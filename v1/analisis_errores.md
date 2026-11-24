# Reporte de Rendimiento y Análisis de Errores — BETO + LoRA vs. TF-IDF + SVM

Todos los datos de este reporte son calculados directamente por `v1/analisis_errores.py` a partir del dataset `data/CPN120.csv` y de las predicciones reales de validación cruzada (5-fold).

## 1. Validación Cruzada (5-Fold Cross Validation)
Resultados obtenidos por fold en `CPN120.csv` (BETO + LoRA):

| Fold | Top-1 Accuracy | Top-5 Accuracy | F1 Macro |
| :--- | :--- | :--- | :--- |
| Fold 0 | 79.30% (`0.7930`) | 90.07% (`0.9007`) | `0.6601` |
| Fold 1 | 79.24% (`0.7924`) | 90.37% (`0.9037`) | `0.6455` |
| Fold 2 | 79.24% (`0.7924`) | 90.71% (`0.9071`) | `0.6503` |
| Fold 3 | 79.18% (`0.7918`) | 91.15% (`0.9115`) | `0.6524` |
| Fold 4 | 78.91% (`0.7891`) | 89.97% (`0.8997`) | `0.6349` |

### Resumen Estadístico (5 Folds):
- **Top-1 Accuracy:** 79.17% ± 0.16% (`0.7917 ± 0.0016`)
- **Top-5 Accuracy:** 90.45% ± 0.48% (`0.9045 ± 0.0048`)
- **F1 Macro:** `0.6486 ± 0.0093`

## 2. Estadísticas del Dataset y Efecto de Cola Larga
- **Tamaño del corpus (`CPN120.csv`):** 31,864 filas, 120 conceptos y **1,785 códigos únicos**.
- **Filas distintas `(Concepto, Respuesta, Codigo)`:** 24,094 (~24.4% de las filas son repeticiones exactas; al normalizar `concepto : respuesta` quedan 20,607 textos únicos).
- **Códigos frecuentes (≥ 100 apariciones):** 43 códigos (2.4% de las clases), encabezados por `angustia` (253), `vestimenta` (226), `miedo` (176), `oficio` (173), `naturaleza` (166), `muerte` (162).
- **Cola larga (< 10 apariciones):** **971 de los 1,785 códigos (54.4%)** aparecen menos de 10 veces en todo el dataset.
- **Soporte crítico (< 5 apariciones):** 677 códigos (37.9%) tienen entre 1 y 4 ejemplos, incluyendo **294 códigos (16.5%)** que aparecen una sola vez (imposibles de acertar cuando caen en el conjunto de test).

## 3. Desempeño por Estrato: BETO + LoRA vs. TF-IDF + SVM

> **Nota metodológica sobre las predicciones (Secciones 3, 4 y 5):** Estos análisis se calculan a partir de los tensores de probabilidad predicha (`.npz`) de una segunda ejecución completa de validación cruzada (5 folds en RX 9070 XT, top-1 promedio 79,13 % y fold 0 = 78,96 %, disponible en `v1/resultados_v1/lora_5folds_corrida2_rx9070xt.csv`). Esto explica variaciones marginales de décimas frente al resumen oficial de la sección 1 (79,17 %).

Comparación calculada sobre las predicciones reales del **Fold 0** (6,373 ejemplos de test) y el acumulado de **5 folds** de BETO + LoRA (31,864 ejemplos):

| Situación en Test | Casos (Fold 0) | LoRA v1 (Fold 0) | TF-IDF + SVM (Fold 0) | LoRA v1 (5 Folds) |
| :--- | :--- | :--- | :--- | :--- |
| Par `(concepto, respuesta)` ya visto en train | 2,871 (45.0%) | **96.3%** | 95.7% | 96.3% (14,402 casos) |
| Respuesta nueva (no vista en train) | 3,502 (55.0%) | **64.8%** | 60.1% | 65.0% (17,462 casos) |
| Código frecuente en train (≥ 100 ejemplos) | 525 (8.2%) | **91.2%** | 90.1% | 90.4% (2,819 casos) |
| Código de frecuencia media (10 a 99 ejemplos) | 4,789 (75.1%) | **81.2%** | 78.2% | 81.7% (24,184 casos) |
| Código raro en train (5 a 9 ejemplos) | 608 (9.5%) | **75.3%** | 72.0% | 70.3% (2,686 casos) |
| Código en cola larga extrema (1 a 4 ejemplos) | 369 (5.8%) | **56.1%** | 52.6% | 56.2% (1,799 casos) |
| Código sin ejemplos en train (0 ejemplos) | 82 (1.3%) | **0.0%** | 0.0% | 0.0% (376 casos) |

### F1-Score Promedio según Frecuencia Total del Código (5 Folds):
- **Códigos con ≥ 100 apariciones (43 clases):** F1 promedio = `0.8501` (85.0%).
- **Códigos con 10 a 99 apariciones (771 clases):** F1 promedio = `0.7784` (77.8%).
- **Códigos con 2 a 9 apariciones (677 clases):** F1 promedio = `0.5502` (55.0%).
- **Códigos con 1 sola aparición (294 clases):** F1 promedio = `0.0000` (0.0%, ya que no pueden aparecer simultáneamente en train y test).

## 4. Principales Confusiones Reales del Clasificador (5 Folds)
Sobre las 31,864 predicciones de test de los 5 folds, BETO + LoRA comete **6,650 errores en Top-1** (20.87%). En **3,596 de esos errores (54.1%)**, el código verdadero sí se encuentra dentro del **Top-5** predicho por el modelo.

Pares `(Código Real → Código Predicho)` más frecuentes en los errores reales de validación cruzada:

| # | Código Real | Código Predicho | Casos | % en Top-5 | Ejemplo Real del Dataset (`Concepto → Respuesta`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `frutas` | `verduras` | 11 | 100% | *cosecha* → "La temporada de cosecha depende de cada fruta o verdura" |
| 2 | `comida` | `alimentarse` | 8 | 100% | *reserva* → "alimento" |
| 3 | `sabor` | `comida_sabrosa` | 8 | 100% | *delicia* → "apetitoso" |
| 4 | `autoridad` | `imposicion` | 7 | 100% | *monarca* → "Mandamás" |
| 5 | `grupo` | `comunidad` | 7 | 100% | *poblado* → "conjunto pequeño" |
| 6 | `malintencionado` | `malefico` | 7 | 100% | *malicia* → "hacer algo encontra del bien" |
| 7 | `material_construccion` | `madera` | 7 | 100% | *cantina* → "Madera" |
| 8 | `autoridad` | `liderazgo` | 6 | 83% | *capitán* → "dirige" |
| 9 | `color_oscuro` | `color_claro` | 6 | 100% | *granito* → "gris oscuro" |
| 10 | `comunidad` | `grupo` | 6 | 83% | *entidad* → "Un grupo de personas" |
| 11 | `identidad` | `subjetivo` | 6 | 83% | *reflejo* → "yo mismo" |
| 12 | `negativo` | `positivo` | 6 | 100% | *emoción* → "pueden ser buenas o malas" |
| 13 | `personalidad` | `caracteristica` | 6 | 67% | *encanto* → "persona encantadora" |
| 14 | `personalidad_multiple` | `personalidad` | 6 | 83% | *facetas* → "lados de la personalidad" |
| 15 | `piedra_valiosa` | `materia_prima` | 6 | 100% | *místico* → "gemas" |

### Patrones Observados en los Errores Reales:
- **Cuasi-sinonimia y solapamiento entre códigos del libro de códigos:** pares como `temor` ↔ `miedo`, `grupo` ↔ `comunidad`, `malintencionado` ↔ `malefico`, `comida` ↔ `alimentarse` o `sabor` ↔ `comida_sabrosa` corresponden a respuestas breves donde ambos códigos son semánticamente cercanos.
- **Categorías co-hipónimas o asociadas:** confusiones como `frutas` ↔ `verduras` o `material_construccion` ↔ `madera` ocurren cuando una misma respuesta (ej. *"verduras"*, *"madera"*) se asocia en el dataset a un código específico en unos conceptos y a uno más general en otros.
- **Polisemia o valencia opuesta sin contexto adicional:** pares como `negativo` ↔ `positivo` o `color_oscuro` ↔ `color_claro` aparecen en respuestas ambiguas o de múltiples palabras (ej. *"pueden ser buenas o malas"*).

## 5. Calibración de Confianza (Probabilidad Top-1)
Relación entre la probabilidad asignada por BETO + LoRA al código Top-1 y su tasa de acierto real (calculada sobre los 5 folds):

| Rango de Confianza (Probabilidad Top-1) | Casos en Test (5 Folds) | % del Total | Accuracy Top-1 Real | Accuracy Top-5 Real |
| :--- | :--- | :--- | :--- | :--- |
| Alta (> 90%) | 23,006 | 72.2% | **93.4%** | 97.3% |
| Media (50% – 90%) | 5,930 | 18.6% | **52.2%** | 81.8% |
| Baja (< 50%) | 2,928 | 9.2% | **21.2%** | 54.1% |

## 6. Conclusiones Respaldadas por los Experimentos
- **Ventaja de BETO + LoRA frente a TF-IDF + SVM en generalización:** mientras ambos modelos rinden ~96% cuando el par `(concepto, respuesta)` ya apareció en entrenamiento, **BETO + LoRA supera a TF-IDF + SVM por ~5 puntos porcentuales en respuestas nuevas no vistas** (64.8% vs. 60.1% en Fold 0) y en F1 Macro global (0.649 vs. 0.608 en 5-fold), gracias a las representaciones contextuales de BETO.
- **Eficiencia de PEFT (LoRA):** se entrena solo el **1.73% de los parámetros** de BETO (1,962,489 de 113,186,034 parámetros), permitiendo completar los 5 folds en hardware local de consumo sin entrenar el modelo base completo (Full Fine-Tuning no fue evaluado en este proyecto).
- **Utilidad del Top-5 y de la probabilidad para codificación asistida:** el salto de ~79.2% (Top-1) a **~90.5% (Top-5)** y la alta precisión cuando la confianza supera el 90% (~93.4% de acierto en el 72.2% de los datos) permiten automatizar los casos de alta confianza y ofrecer 5 candidatos sugeridos (`v1/predict.py`) al codificador humano en los casos ambiguos o de cola larga.
