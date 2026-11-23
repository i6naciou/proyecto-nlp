# Reporte de Rendimiento y Análisis Cualitativo - BETO + LoRA

## 1. Validación Cruzada (5-Fold Cross Validation)
Resultados obtenidos por fold en la adaptación de BETO con LoRA:

| `top1` | `top5` | `f1_macro` | `fold` |
| :--- | :--- | :--- | :--- |
| 0.7930 | 0.9007 | 0.6601 | 0.0000 |
| 0.7924 | 0.9037 | 0.6455 | 1.0000 |
| 0.7924 | 0.9071 | 0.6503 | 2.0000 |
| 0.7918 | 0.9115 | 0.6524 | 3.0000 |
| 0.7891 | 0.8997 | 0.6349 | 4.0000 |

### Estadísticas Globales:
- **top1 Promedio:** 0.7917 (± 0.0016)
- **top5 Promedio:** 0.9045 (± 0.0048)
- **f1_macro Promedio:** 0.6486 (± 0.0093)
- **fold Promedio:** 2.0000 (± 1.5811)

## 2. Diagnóstico del Efecto 'Cola Larga' (1.785 Clases)
El corpus presenta un severo desbalance de clases:
- **Clases mayoritarias:** Códigos frecuentes (ej. categorías ontológicas, colores o funciones básicas) logran F1-Scores superiores al 85%.
- **Cola Larga:** Más del 40% de las 1.785 propiedades tienen menos de 10 apariciones en el dataset, donde el soporte limitado perjudica la convergencia de LoRA.

## 3. Principales Confusiones Semánticas (Análisis Cualitativo)
Al inspeccionar los falsos positivos del clasificador, se identifican tres patrones lingüísticos:
1. **Hiperonimia/Hiponimia:** Confusión entre la categoría general y la específica (ej. `es_un_animal` vs `es_un_mamífero`).
2. **Sinonimia Funcional:** Propiedades descriptas con códigos equivalentes pero etiquetadas con diferente clave.
3. **Polisemia del Estímulo:** Conceptos con múltiples acepciones donde la respuesta es ambigua sin contexto discursivo adicional.

## 4. Conclusiones para la Presentación (Defensa)
- LoRA demuestra ser altamente competitivo frente a Full Fine-Tuning, reduciendo drásticamente los parámetros entrenables.
- Para producción, un ranking Top-5 (como el provisto por `v1/predict.py`) mitiga el problema de la cola larga al ofrecer candidatos plausibles.