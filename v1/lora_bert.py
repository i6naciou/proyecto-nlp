"""
v1: BETO + LoRA, mejorado respecto a la v0.

Cambios respecto a v0 (y por qué):
  1. Texto normalizado (minúsculas, sin tildes ni símbolos), como el mejor método del paper.
  2. Evaluación con 5-fold (igual que el paper) y métricas top-1, top-5 y F1 macro.
  3. Más épocas (10 en vez de 5): en v0 la accuracy seguía subiendo en la época 5.
  4. Tasa de aprendizaje separada y más alta para la capa clasificadora (parte desde cero
     y tiene que aprender 1.785 clases), y más baja para los adaptadores LoRA.
  5. Ciclo de entrenamiento propio en PyTorch con los textos tokenizados una sola vez
     al inicio, en vez de Trainer (más rápido y más fácil de explicar).
  6. Guarda las probabilidades de cada fold, para combinar después con TF-IDF.

Uso:
    python lora_bert.py --csv ../data/CPN120.csv                 # los 5 folds
    python lora_bert.py --csv ../data/CPN120.csv --folds 0       # solo el fold 0 (prueba)
"""

import argparse
import os
import time

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from comun import cargar_datos, metricas, particiones, resumir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="../data/CPN120.csv")
    p.add_argument("--modelo", default="dccuchile/bert-base-spanish-wwm-cased")
    p.add_argument("--salida", default="resultados_v1")
    p.add_argument("--folds", default="0,1,2,3,4", help="folds a correr, ej. '0' o '0,1,2,3,4'")
    p.add_argument("--epocas", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr_lora", type=float, default=5e-4)
    p.add_argument("--lr_clasificador", type=float, default=3e-3)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--max_len", type=int, default=32)
    p.add_argument("--semilla", type=int, default=42)
    return p.parse_args()


def crear_modelo(args, n_clases, dispositivo):
    modelo = AutoModelForSequenceClassification.from_pretrained(args.modelo, num_labels=n_clases)
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.1,
        target_modules=["query", "value"],
    )
    modelo = get_peft_model(modelo, cfg)
    return modelo.to(dispositivo)


def entrenar_fold(args, modelo, dl_train, dispositivo):
    # Dos grupos de parámetros: capa clasificadora (lr alta) y adaptadores LoRA (lr baja)
    params_clf = [p for n, p in modelo.named_parameters() if p.requires_grad and "classifier" in n]
    params_lora = [p for n, p in modelo.named_parameters() if p.requires_grad and "classifier" not in n]
    optim = torch.optim.AdamW(
        [
            {"params": params_lora, "lr": args.lr_lora},
            {"params": params_clf, "lr": args.lr_clasificador},
        ],
        weight_decay=0.01,
    )
    pasos = len(dl_train) * args.epocas
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * pasos), pasos)
    usar_amp = dispositivo == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)

    modelo.train()
    for ep in range(args.epocas):
        t0, perdida_total = time.time(), 0.0
        for ids, mask, y in dl_train:
            ids, mask, y = ids.to(dispositivo), mask.to(dispositivo), y.to(dispositivo)
            with torch.autocast(device_type=dispositivo, dtype=torch.float16, enabled=usar_amp):
                perdida = modelo(input_ids=ids, attention_mask=mask, labels=y).loss
            optim.zero_grad()
            scaler.scale(perdida).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            perdida_total += perdida.item()
        print(f"  época {ep + 1}/{args.epocas}  pérdida={perdida_total / len(dl_train):.4f}"
              f"  ({time.time() - t0:.0f} s)")


@torch.no_grad()
def predecir(modelo, dl, dispositivo):
    modelo.eval()
    salidas = []
    for ids, mask, _ in dl:
        with torch.autocast(device_type=dispositivo, dtype=torch.float16, enabled=dispositivo == "cuda"):
            logits = modelo(input_ids=ids.to(dispositivo), attention_mask=mask.to(dispositivo)).logits
        salidas.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(salidas)


def main():
    args = parse_args()
    torch.manual_seed(args.semilla)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {dispositivo}")

    df = cargar_datos(args.csv)
    clases = sorted(df["Codigo"].unique())
    codigo2id = {c: i for i, c in enumerate(clases)}
    y = torch.tensor(df["Codigo"].map(codigo2id).values)
    print(f"Ejemplos: {len(df)} | Clases: {len(clases)}")

    # Tokenizar TODO una sola vez (ahorra tiempo en cada época y cada fold)
    tok = AutoTokenizer.from_pretrained(args.modelo)
    enc = tok(list(df["texto"]), truncation=True, max_length=args.max_len,
              padding="longest", return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]

    folds_a_correr = {int(f) for f in args.folds.split(",")}
    os.makedirs(args.salida, exist_ok=True)
    resultados = []

    for fold, tr, te in particiones(df):
        if fold not in folds_a_correr:
            continue
        print(f"\n===== Fold {fold} (train={len(tr)}, test={len(te)}) =====")
        torch.manual_seed(args.semilla + fold)
        dl_train = DataLoader(TensorDataset(ids[tr], mask[tr], y[tr]), batch_size=args.batch, shuffle=True)
        dl_test = DataLoader(TensorDataset(ids[te], mask[te], y[te]), batch_size=args.batch * 4)

        modelo = crear_modelo(args, len(clases), dispositivo)
        if fold == min(folds_a_correr):
            modelo.print_trainable_parameters()
        entrenar_fold(args, modelo, dl_train, dispositivo)

        probs = predecir(modelo, dl_test, dispositivo)
        r = metricas(df.loc[te, "Codigo"].values, probs, clases)
        r["fold"] = fold
        print(f"  Resultado fold {fold}: {r}")
        resultados.append(r)

        # Guardar probabilidades del fold (para combinar con TF-IDF después)
        np.savez_compressed(os.path.join(args.salida, f"lora_probs_fold{fold}.npz"),
                            indices=te, probs=probs.astype(np.float16))
        # Guardar el modelo entrenado para futuras predicciones
        ruta_checkpoint = os.path.join(args.salida, f"checkpoint_fold_{fold}")
        modelo.save_pretrained(ruta_checkpoint)
        tok.save_pretrained(ruta_checkpoint)
        del modelo
        if dispositivo == "cuda":
            torch.cuda.empty_cache()

    nombre = "lora_folds_" + args.folds.replace(",", "-") + ".csv"
    resumir(resultados).to_csv(os.path.join(args.salida, nombre), index=False)


if __name__ == "__main__":
    main()
