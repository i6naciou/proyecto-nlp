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
| Evaluación 5-fold con top-1, top-5 y F1 macro | Es como evalúa el paper; permite una comparación justa |
| Normalización del texto (minúsculas, sin tildes ni símbolos, sin quitar stop words) | Mejor preprocesamiento según los autores; reduce ruido como "Condimentos" o "tecnologíco" |
| Versión simplificada TF-IDF + SVM | El enunciado pide comparar con versiones simplificadas |
| LoRA: 10 épocas en vez de 5 | En v0 el modelo seguía mejorando al terminar |
| LoRA: tasa de aprendizaje mayor para la capa clasificadora | Esa capa parte desde cero y tiene 1.785 clases |
| LoRA: ciclo de entrenamiento propio, textos tokenizados una vez | Más rápido y más simple de explicar |

**Cómo correrlo.**
```
cd v1
python tfidf_svm.py --csv ../data/CPN120.csv
python lora_bert.py --csv ../data/CPN120.csv
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
| **v1: TF-IDF + SVM** | 5-fold | 76,4 % ± 0,6 | 87,0 % |
| **v1: BETO + LoRA** | 5-fold | *pendiente* | *pendiente* |

El fine-tuning de GPT-3.5 del paper costó US$87,77. Nuestros modelos corren localmente y gratis
en una GTX 1650.
