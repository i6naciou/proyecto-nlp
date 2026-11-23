# Proyecto NLP 2026-1 — PEFT-Clasificación (Lang-Chi)

Clasificación automática de propiedades semánticas (dataset **CPN120**): dado un concepto y la
respuesta de un participante (ej. *caracol → "baboso"*), predecir el código asignado por
codificadores humanos (ej. `baboso`). Hay **1.785 códigos posibles**.

Paper de referencia: Ramos, D., Chaigneau, S., Moreno, S., Canessa, E. (2025).
*Using Chat-GPT for coding properties in semantic memory studies.*

## Estructura

```
data/CPN120.csv          datos entregados por el curso
v0_cruda/                versión cruda generada por IA
v1/                      versión mejorada (5-fold, normalización, comparación con TF-IDF)
```

## Datos

- 31.864 filas, 120 conceptos, 1.785 códigos.
- 24.094 filas distintas: ~24 % son repeticiones (varias personas dieron la misma respuesta).
- El paper reporta 27.138 propiedades para CPN120, con los mismos 1.785 códigos. Desconocemos
  el origen de la diferencia (posiblemente otra versión o limpieza del dataset).

---

## v0 — Versión cruda generada por IA

**Cómo se generó (prompt).** No se escribió un prompt único: la v0 salió de una conversación
con una IA (Claude, de Anthropic). Se le entregó:

1. El enunciado del proyecto (PDF).
2. El archivo `CPN120.csv`.
3. Las características del PC donde se iba a entrenar (GPU GTX 1650 de 4 GB, 16 GB RAM, Windows 10).

y se le pidió ayuda para resolver el proyecto. **No se le entregó el paper de referencia**, así
que la IA no sabía cómo evalúa el paper (5-fold, top-5).

**Nota sobre "crudo".** La IA probó el script antes de entregarlo, detectó un error de
compatibilidad (el parámetro `warmup_ratio` fue eliminado en `transformers` 5) y lo corrigió.
Fuera de eso, `v0_cruda/entrenar_lora.py` es exactamente lo que entregó la IA, **sin
modificaciones del equipo**.

**Qué hace.**
- Modelo base: BETO (`dccuchile/bert-base-spanish-wwm-cased`), BERT en español.
- PEFT con **LoRA** (r = 16) en las capas `query` y `value` de la atención. Se congela BETO
  completo y solo se entrenan los adaptadores LoRA y la capa clasificadora:
  **1.962.489 de 113.186.034 parámetros (1,73 %)**.
- Una sola división aleatoria 80/10/10 (train/validación/test), 5 épocas.

**Resultados v0 (test, 3.187 ejemplos).**

| Métrica | Valor |
|---|---|
| Accuracy | 66,2 % |
| F1 macro | 0,465 |
| F1 ponderado | 0,620 |

La accuracy en validación seguía subiendo al terminar la época 5 (38 → 55 → 62 → 65 → 66 %):
el modelo quedó sub-entrenado.

**Cómo correrlo.**
```
cd v0_cruda
python entrenar_lora.py --csv ../data/CPN120.csv
```

---

## v1 — Mejoras

Uso de IA: la IA propuso cambios y el equipo decidió cuáles aplicar y por qué.

| Cambio | Por qué |
|---|---|
| Particiones 5-fold (las del paper) y métricas top-1, top-5 y F1 macro | Es como evalúa el paper; permite una comparación justa. TF-IDF y LoRA se evaluaron en los 5 folds completos (fold 0 en GTX 1650 y folds 1 al 4 en RTX 4070 Super en ~10 min con FP16). La varianza entre folds fue mínima (± 0,2 % en Top-1) |
| Normalización del texto (minúsculas, sin tildes ni símbolos, sin quitar stop words) | Mejor preprocesamiento según los autores; reduce ruido como "Condimentos" o "tecnologíco" |
| Versión simplificada TF-IDF + SVM | El enunciado pide comparar con versiones simplificadas |
| LoRA: 10 épocas en vez de 5 | En v0 el modelo seguía mejorando al terminar |
| LoRA: tasa de aprendizaje mayor para la capa clasificadora | Esa capa parte desde cero y tiene 1.785 clases |
| LoRA: ciclo de entrenamiento propio en PyTorch, textos tokenizados una vez | Más simple de explicar que `Trainer` |

**Cómo correrlo.**
```
cd v1
python tfidf_svm.py --csv ../data/CPN120.csv
python lora_bert.py --csv ../data/CPN120.csv             # 5 folds completos
python lora_bert.py --csv ../data/CPN120.csv --folds 0   # solo fold 0
```

---

## Comparación con el paper (CPN120, accuracy en test)

| Sistema | Evaluación | Top-1 | Top-5 |
|---|---|---|---|
| AC-PLT (paper, leído de la Figura 4) | 5-fold | ~54 % | ~62 % |
| GPT-3.5 fine-tuning (paper) | 5-fold | 58,2 % | — |
| GPT-4o one-shot (paper) | 5-fold | 17,8 % | — |
| GPT-3.5 one-shot (paper) | 5-fold | 7,3 % | — |
| **v0: BETO + LoRA** | 1 división 80/10/10 | 66,2 % | — |
| **v1: TF-IDF + SVM** | 5-fold (promedio) | 76,4 % ± 0,6 | 87,0 % |
| **v1: TF-IDF + SVM** | solo fold 0 | 76,1 % | 86,8 % |
| **v1: BETO + LoRA** | **5-fold (promedio)** | **79,2 % ± 0,2** | **90,5 % ± 0,5** |
| **v1: BETO + LoRA** | solo fold 0 | 79,3 % | 90,1 % |

Resultados de LoRA v1 por fold (`v1/resultados_v1/lora_5folds_completo.csv`):
- Fold 0: 79,3 % Top-1 | 90,1 % Top-5 | F1 macro 0,660
- Fold 1: 79,2 % Top-1 | 90,4 % Top-5 | F1 macro 0,645
- Fold 2: 79,2 % Top-1 | 90,7 % Top-5 | F1 macro 0,650
- Fold 3: 79,2 % Top-1 | 91,2 % Top-5 | F1 macro 0,652
- Fold 4: 78,9 % Top-1 | 90,0 % Top-5 | F1 macro 0,635
- **Promedio ± std:** **79,2 % ± 0,2** (Top-1) | **90,5 % ± 0,5** (Top-5) | **0,649 ± 0,009** (F1 macro)

F1 macro en 5-fold: TF-IDF ~0,615 vs. LoRA v1 **0,649 ± 0,009**, confirmando que LoRA clasifica mejor los códigos poco frecuentes en todas las particiones.

**Sobre el sobreajuste.** La pérdida de entrenamiento de LoRA v1 bajó hasta 0,13 en la época 10,
lo que hacía sospechar memorización (el paper reporta algo así con GPT-3.5 en CPN27: 91 % en
train vs. 67 % en test). Sin embargo, en test LoRA v1 promedia 79,2 % en los 5 folds, superando consistentemente a la v0 y a
TF-IDF: la pérdida baja se explica en buena parte por las respuestas repetidas del dataset.

Tiempos de entrenamiento:
- Fold 0 en GTX 1650 (4 GB): ~13 min por época (~2 h 15 min total).
- Folds 1 al 4 en RTX 4070 Super (12 GB, FP16): ~15–18 s por época (~2,5 min por fold, ~10 min en total).
- Folds 0 al 4 en RX 9070 XT (16 GB, RDNA 4, ROCm 10.2 en Windows): ~7–8 s por época (~1 min 15 s por fold, ~6 min 15 s los 5 folds completos).
- Folds 0 al 4 en RTX 3060 Laptop (6 GB): ~30–39 s por época (~5–6 min por fold) (~30 min en total).

El fine-tuning de GPT-3.5 del paper costó US$87,77. Nuestros modelos corren localmente y gratis
en hardware de consumo.
