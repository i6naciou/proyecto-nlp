"""
Versión 0 (cruda, generada por IA): clasificación de propiedades semánticas con PEFT (LoRA).

Tarea: dado un concepto y la respuesta que dio un participante (ej. "caracol" + "baboso"),
predecir el código asignado por los codificadores humanos (ej. "baboso").

Modelo base: BETO (BERT en español, dccuchile/bert-base-spanish-wwm-cased).
PEFT: LoRA sobre las capas de atención (query, value). Solo se entrenan los adaptadores
LoRA y la capa de clasificación; el resto del modelo queda congelado.

Uso:
    python entrenar_lora.py --csv ../data/CPN120.csv
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="../data/CPN120.csv")
    p.add_argument("--modelo", default="dccuchile/bert-base-spanish-wwm-cased")
    p.add_argument("--salida", default="resultados_v0")
    p.add_argument("--epocas", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max_len", type=int, default=32)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--semilla", type=int, default=42)
    return p.parse_args()


def fijar_semilla(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class DatasetPropiedades(Dataset):
    def __init__(self, textos, etiquetas, tokenizer, max_len):
        self.enc = tokenizer(list(textos), truncation=True, max_length=max_len)
        self.etiquetas = list(etiquetas)

    def __len__(self):
        return len(self.etiquetas)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.etiquetas[i]
        return item


def calcular_metricas(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def main():
    args = parse_args()
    fijar_semilla(args.semilla)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {dispositivo}")

    # 1. Cargar datos
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["Concepto", "Respuesta", "Codigo"])
    df["texto"] = df["Concepto"].str.strip() + ": " + df["Respuesta"].str.strip()

    # 2. Codificar etiquetas
    codigos = sorted(df["Codigo"].unique())
    codigo2id = {c: i for i, c in enumerate(codigos)}
    id2codigo = {i: c for c, i in codigo2id.items()}
    df["label"] = df["Codigo"].map(codigo2id)
    print(f"Ejemplos: {len(df)} | Clases: {len(codigos)}")

    # 3. División train / validación / test (80/10/10)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=args.semilla)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.semilla)

    # 4. Tokenizer y modelo base
    tokenizer = AutoTokenizer.from_pretrained(args.modelo)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        args.modelo,
        num_labels=len(codigos),
        id2label=id2codigo,
        label2id=codigo2id,
    )

    # 5. Configuración LoRA (PEFT)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.1,
        target_modules=["query", "value"],
    )
    modelo = get_peft_model(modelo, lora_cfg)
    modelo.print_trainable_parameters()

    ds_train = DatasetPropiedades(train_df["texto"], train_df["label"], tokenizer, args.max_len)
    ds_val = DatasetPropiedades(val_df["texto"], val_df["label"], tokenizer, args.max_len)
    ds_test = DatasetPropiedades(test_df["texto"], test_df["label"], tokenizer, args.max_len)

    # 6. Entrenamiento
    # Warmup: 6% de los pasos totales (se calcula a mano porque transformers 5 eliminó warmup_ratio)
    pasos_totales = int(np.ceil(len(ds_train) / args.batch)) * args.epocas
    targs = TrainingArguments(
        output_dir=args.salida,
        num_train_epochs=args.epocas,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=int(0.06 * pasos_totales),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        fp16=(dispositivo == "cuda"),
        report_to="none",
        seed=args.semilla,
    )
    trainer = Trainer(
        model=modelo,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=calcular_metricas,
    )
    trainer.train()

    # 7. Evaluación final en test
    res = trainer.evaluate(ds_test, metric_key_prefix="test")
    print("\n=== Resultados en TEST ===")
    for k, v in res.items():
        print(f"{k}: {v}")

    # 8. Guardar predicciones y adaptador
    pred = trainer.predict(ds_test)
    test_df = test_df.copy()
    test_df["prediccion"] = [id2codigo[i] for i in np.argmax(pred.predictions, axis=-1)]
    os.makedirs(args.salida, exist_ok=True)
    test_df[["Concepto", "Respuesta", "Codigo", "prediccion"]].to_csv(
        os.path.join(args.salida, "predicciones_test.csv"), index=False
    )
    modelo.save_pretrained(os.path.join(args.salida, "adaptador_lora"))
    tokenizer.save_pretrained(os.path.join(args.salida, "adaptador_lora"))
    print(f"\nListo. Resultados guardados en {args.salida}/")


if __name__ == "__main__":
    main()
