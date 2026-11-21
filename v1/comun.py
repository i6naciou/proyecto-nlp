"""
Funciones compartidas por todos los experimentos v1.

- Normalización del texto (igual que el mejor método del paper de Ramos et al. 2025:
  minúsculas + quitar caracteres especiales, SIN quitar stop words).
- Validación cruzada de 5 particiones (5-fold), igual que el paper.
- Métricas: top-1, top-5 y F1 macro.
"""

import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


def normalizar(texto: str) -> str:
    """'Implica una responsbailidad!!' -> 'implica una responsbailidad'"""
    texto = str(texto).lower().strip()
    # quitar tildes (así 'tecnologíco' y 'tecnologico' quedan iguales)
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    # quitar todo lo que no sea letra, número o espacio
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    df = pd.read_csv(ruta_csv).dropna(subset=["Concepto", "Respuesta", "Codigo"])
    df = df.reset_index(drop=True)
    df["texto"] = df["Concepto"].map(normalizar) + " : " + df["Respuesta"].map(normalizar)
    return df


def particiones(df: pd.DataFrame, n_folds: int = 5, semilla: int = 42):
    """Entrega (fold, indices_train, indices_test). Mismas particiones para todos los modelos."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=semilla)
    for i, (tr, te) in enumerate(kf.split(df)):
        yield i, tr, te


def metricas(y_true, puntajes, clases):
    """
    y_true:   lista de códigos verdaderos (texto)
    puntajes: matriz (n_ejemplos x n_clases) con el puntaje de cada clase
    clases:   nombre de la clase de cada columna de 'puntajes'
    """
    clases = np.asarray(clases)
    y_true = np.asarray(y_true)
    orden = np.argsort(-puntajes, axis=1)
    top1 = clases[orden[:, 0]]
    top5 = clases[orden[:, :5]]
    return {
        "top1": float(np.mean(top1 == y_true)),
        "top5": float(np.mean([y in fila for y, fila in zip(y_true, top5)])),
        "f1_macro": float(f1_score(y_true, top1, average="macro", zero_division=0)),
    }


def resumir(resultados_por_fold):
    df = pd.DataFrame(resultados_por_fold)
    print("\nPor fold:")
    print(df.round(4).to_string(index=False))
    print("\nPromedio ± desviación:")
    for col in ["top1", "top5", "f1_macro"]:
        print(f"  {col}: {df[col].mean():.4f} ± {df[col].std():.4f}")
    return df
