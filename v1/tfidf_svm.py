"""
Versión simplificada: TF-IDF (n-gramas de caracteres) + SVM lineal, evaluada con 5-fold.
Corre en segundos y sirve de punto de comparación para LoRA.

Uso:
    python tfidf_svm.py --csv ../data/CPN120.csv
"""

import argparse
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from comun import cargar_datos, metricas, particiones, resumir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="../data/CPN120.csv")
    p.add_argument("--salida", default="resultados_v1")
    args = p.parse_args()

    df = cargar_datos(args.csv)
    print(f"Ejemplos: {len(df)} | Clases: {df['Codigo'].nunique()}")

    resultados = []
    for fold, tr, te in particiones(df):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        X_tr = vec.fit_transform(df.loc[tr, "texto"])
        X_te = vec.transform(df.loc[te, "texto"])
        modelo = LinearSVC(C=1.0).fit(X_tr, df.loc[tr, "Codigo"])
        r = metricas(df.loc[te, "Codigo"], modelo.decision_function(X_te), modelo.classes_)
        r["fold"] = fold
        print(f"fold {fold}: {r}")
        resultados.append(r)

    os.makedirs(args.salida, exist_ok=True)
    resumir(resultados).to_csv(os.path.join(args.salida, "tfidf_svm_folds.csv"), index=False)


if __name__ == "__main__":
    main()
