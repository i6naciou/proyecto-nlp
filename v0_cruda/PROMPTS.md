# Generación de la v0 con IA

La v0 (`entrenar_lora.py`) se generó **exclusivamente mediante prompt engineering conversacional**
(5 mensajes) con una IA (Claude, de Anthropic). El equipo no escribió código en esta etapa.

## 1. Mensajes enviados a la IA

Transcripción de los mensajes. Se corrigieron la ortografía y la puntuación y se omitieron frases
personales [...]; el contenido de cada solicitud no se modificó.

**Mensaje 1** (adjuntos: enunciado del proyecto en PDF y `CPN120.csv`):
> Necesito realizar este trabajo. ¿Puedes indicarme primero qué debo tener instalado en mi PC y
> luego ayudarme a desarrollarlo? [...]

**Mensajes 2 a 4** (contexto de hardware):
> Tengo una GPU GTX 1650.
>
> 16 GB de RAM, procesador Ryzen 5 3600.
>
> Windows 10.

**Mensaje 5:**
> Entrégame una versión que funcione, junto con todos los comandos de PowerShell para instalar lo
> necesario.

## 2. Qué hizo la IA entre los mensajes

- Analizó el dataset por su cuenta (tamaño, número de códigos, distribución) y calculó un baseline
  TF-IDF de referencia (~76 %).
- Antes de entregar el script, lo probó con un modelo de prueba diminuto y detectó un error de
  compatibilidad: el parámetro `warmup_ratio` fue eliminado en `transformers` 5.x.
- Informó el error y ofreció entregar la versión con el error o una corregida. El equipo pidió la
  versión funcional (mensaje 5), y la IA reemplazó `warmup_ratio` por un cálculo equivalente con
  `warmup_steps`.
- Ese es el único cambio respecto a lo generado originalmente. El equipo no modificó el código.

**Lo que la IA no tenía:** el paper de referencia. Por eso la v0 evalúa con una sola división
aleatoria, sin top-5 y sin normalizar el texto, a diferencia del paper.

---

## 3. Prompt recomendado (reconstruido a posteriori)

> **Este NO es el prompt con que se generó la v0.** Es el prompt que, con lo aprendido en el
> proyecto, habría producido directamente una solución equivalente a la v1. Lo incluimos como
> ejercicio de prompt engineering.

```text
ROL
Actúa como ingeniero de Machine Learning especializado en NLP en español.

CONTEXTO
Proyecto universitario (curso de NLP) para la start-up Lang-Chi. Debemos replicar y
mejorar el problema de Ramos et al. (2025), "Using Chat-GPT for coding properties in
semantic memory studies": asignar automáticamente códigos a propiedades semánticas.

DATOS
CSV "CPN120.csv" con columnas Concepto, Respuesta, Codigo.
- 31.864 filas, 120 conceptos, 1.785 códigos distintos (cola larga: 294 códigos
  aparecen una sola vez). ~24 % de filas son repetidas.
- Ejemplo: Concepto="caracol", Respuesta="baboso", Codigo="baboso".
- Las respuestas son muy cortas (1–5 palabras), con faltas de ortografía y mayúsculas.

RESTRICCIONES
- Solo modelos locales (nada de APIs pagadas).
- Hardware: Windows 10, GPU NVIDIA GTX 1650 de 4 GB, 16 GB de RAM, Python 3.11.
- Librerías: torch (CUDA), transformers 5.x, peft 0.22, scikit-learn, pandas.
  Ten en cuenta que transformers 5 eliminó parámetros antiguos (por ejemplo
  warmup_ratio y evaluation_strategy): no los uses.

TAREA
1. Clasificador con PEFT: LoRA sobre BETO (dccuchile/bert-base-spanish-wwm-cased),
   r=16, sobre las capas query y value. Usa una tasa de aprendizaje más alta para la
   capa clasificadora que para LoRA, porque parte desde cero con 1.785 clases.
   Entrena al menos 10 épocas.
2. Versión simplificada para comparar: TF-IDF con n-gramas de caracteres (2–5) + LinearSVC.
3. Evaluación igual que el paper: validación cruzada de 5 particiones (KFold, shuffle,
   semilla 42, las mismas particiones para ambos modelos), con métricas top-1, top-5
   y F1 macro (promedio ± desviación estándar).
4. Preprocesamiento como el paper: minúsculas, sin tildes ni caracteres especiales,
   SIN quitar stop words. Entrada al modelo: "concepto : respuesta".
5. Eficiencia en 4 GB: FP16, batch 64, max_len 32 y relleno (padding) recortado por
   lote, no al largo máximo del dataset.
6. Permite correr solo algunos folds (--folds 0) y guarda los resultados de cada
   corrida en un archivo distinto, para no sobrescribirlos.

REFERENCIA A SUPERAR (CPN120, paper)
AC-PLT top-1 ≈ 54 %, top-5 ≈ 62 %; GPT-3.5 con fine-tuning 58,2 %; GPT-4o one-shot 17,8 %.

VERIFICACIÓN ANTES DE ENTREGAR
- Ejecuta una prueba rápida de punta a punta (subconjunto de datos y un modelo
  diminuto) para comprobar que todo corre sin errores con las versiones indicadas.
- Si encuentras un error, corrígelo y dime cuál fue y cómo lo arreglaste.
- No inventes resultados: entrega solo los que hayas ejecutado de verdad y deja el
  resto como "pendiente".

FORMATO DE SALIDA
- Archivos: comun.py (datos, normalización, folds, métricas), tfidf_svm.py, lora_bert.py.
- Comentarios en español que expliquen cada bloque del código.
- Comandos de PowerShell para instalar y ejecutar, y tiempo estimado por época.
- Una lista de supuestos que hiciste.
```

### Qué le faltó a la conversación original y por qué importa

| Faltó en la v0 | Consecuencia | Lo resuelve en el prompt |
|---|---|---|
| El paper y su forma de evaluar | Una sola división, sin top-5: no comparable | TAREA, punto 3 |
| Normalización del texto | Ruido ("Condimentos", "tecnologíco") | TAREA, punto 4 |
| Versiones de librerías | Error con `warmup_ratio` | RESTRICCIONES |
| Lr separada y más épocas | v0 sub-entrenada (66 %) | TAREA, punto 1 |
| Padding por lote | Entrenamiento más lento en la GTX 1650 | TAREA, punto 5 |
| Pedir prueba y corrección explícitas | El error se detectó solo porque la IA probó por iniciativa propia | VERIFICACIÓN |
