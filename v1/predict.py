import argparse
import os
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

MODELO_BASE = "dccuchile/bert-base-spanish-wwm-uncased"

def obtener_etiquetas(ruta_csv="data/dataset.csv"):
    """
    Obtiene la lista ordenada de clases únicas para mapear el ID al código real.
    Ajusta la ruta al CSV de datos del repo.
    """
    if os.path.exists(ruta_csv):
        df = pd.read_csv(ruta_csv)
        # Ajustar el nombre de la columna según el CSV (ej: 'codigo', 'label', o 'property_code')
        columna_etiqueta = "codigo" if "codigo" in df.columns else df.columns[-1]
        clases = sorted(df[columna_etiqueta].dropna().unique())
        return {i: str(c) for i, c in enumerate(clases)}
    return None

def main():
    parser = argparse.ArgumentParser(description="Inferencia interactiva con BETO + LoRA")
    parser.add_argument("--concepto", type=str, required=True, help="Ejemplo: caracol")
    parser.add_argument("--respuesta", type=str, required=True, help="Ejemplo: baboso")
    parser.add_argument("--checkpoint", type=str, default="v1/checkpoints/fold_0", help="Ruta al adaptador LoRA")
    parser.add_argument("--top_k", type=int, default=5, help="Cantidad de predicciones a mostrar")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo detectado: {device}")

    # 1. Cargar Tokenizador
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)

    # 2. Obtener nombres de las 1785 clases
    id2label = obtener_etiquetas()
    num_labels = len(id2label) if id2label else 1785

    # 3. Cargar Modelo Base
    print("[*] Cargando arquitectura base...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BASE,
        num_labels=num_labels
    )

    # 4. Acoplar pesos LoRA si existen
    if os.path.exists(args.checkpoint):
        print(f"[*] Cargando adaptadores LoRA desde {args.checkpoint}...")
        model = PeftModel.from_pretrained(model, args.checkpoint)
    else:
        print(f"[!] No se encontró '{args.checkpoint}'. Se ejecutará en modo demostración.")

    model.to(device)
    model.eval()

    # 5. Formatear la entrada igual que en el entrenamiento
    texto_entrada = f"concepto: {args.concepto} | respuesta: {args.respuesta}"
    
    inputs = tokenizer(
        texto_entrada,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    ).to(device)

    # 6. Inferencia y Top-K Softmax
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilidades = F.softmax(logits, dim=-1)[0]

    top_probs, top_indices = torch.topk(probabilidades, k=args.top_k)

    print(f"\n==========================================")
    print(f" Entrada: Concepto='{args.concepto}' | Respuesta='{args.respuesta}'")
    print(f"==========================================")
    for rank, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist()), start=1):
        nombre_clase = id2label[idx] if id2label else f"Clase ID {idx}"
        print(f" {rank}. Código: {nombre_clase:<20} Confianza: {prob * 100:.2f}%")
    print(f"==========================================\n")

if __name__ == "__main__":
    main()