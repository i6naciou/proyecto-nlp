
"""
Predicciones con BETO + LoRA entrenado.

Ejemplo:
python predict.py --concepto "granito" --respuesta "rugoso"
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from comun import cargar_datos, normalizar


MODELO_BASE = "dccuchile/bert-base-spanish-wwm-cased"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concepto", required=True)
    parser.add_argument("--respuesta", required=True)
    parser.add_argument("--csv", default="../data/CPN120.csv")
    parser.add_argument(
        "--checkpoint",
        default="prueba_inferencia/checkpoint_fold_0"
    )
    parser.add_argument("--top_k", type=int, default=5)

    args = parser.parse_args()

    # Comprobar que existe el modelo entrenado
    if not os.path.isdir(args.checkpoint):
        parser.error(
            f"No se encontró el checkpoint: {args.checkpoint}"
        )

    # Recuperar las clases en el mismo orden del entrenamiento
    df = cargar_datos(args.csv)
    clases = sorted(df["Codigo"].unique())
    num_clases = len(clases)

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Dispositivo: {dispositivo}")
    print(f"Categorías disponibles: {num_clases}")
    print("Cargando modelo entrenado...")

    # Cargar el mismo tokenizador usado durante el entrenamiento
    tokenizador = AutoTokenizer.from_pretrained(args.checkpoint)

    # Reproducir la inicialización utilizada en el fold 0
    torch.manual_seed(42)

    modelo_base = AutoModelForSequenceClassification.from_pretrained(
    MODELO_BASE,
    num_labels=num_clases
)

    # Incorporar los pesos LoRA y el clasificador entrenado
    modelo = PeftModel.from_pretrained(
        modelo_base,
        args.checkpoint
    )

    modelo = modelo.to(dispositivo)
    modelo.eval()

    # Aplicar exactamente la normalización de comun.py
    texto = (
        normalizar(args.concepto)
        + " : "
        + normalizar(args.respuesta)
    )

    # Tokenizar como en lora_bert.py
    entradas = tokenizador(
        texto,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )

    entradas = {
        clave: valor.to(dispositivo)
        for clave, valor in entradas.items()
    }

    # Inferencia sin entrenamiento
    with torch.no_grad():
        logits = modelo(**entradas).logits
        probabilidades = torch.softmax(
            logits.float(), dim=-1
        )[0]

    # Obtener las categorías con mayor probabilidad
    cantidad = min(max(1, args.top_k), num_clases)

    valores, indices = torch.topk(
        probabilidades,
        cantidad
    )

    print("\n===== RESULTADOS =====")
    print(f"Concepto: {args.concepto}")
    print(f"Respuesta: {args.respuesta}")
    print("\nCategorías más probables:")

    for posicion, (indice, probabilidad) in enumerate(
        zip(indices.tolist(), valores.tolist()),
        start=1
    ):
        categoria = clases[indice]
        porcentaje = probabilidad * 100

        print(
            f"{posicion}. {categoria}: {porcentaje:.2f}%"
        )


if __name__ == "__main__":
    main()
