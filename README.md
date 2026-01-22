# Emotion Analysis Project

Proyecto de análisis de emociones enfocado en generar **datasets sintéticos progresivamente más realistas** para evaluación de modelos de machine learning.

⚠️ **IMPORTANTE: Todos los datos en este proyecto son completamente artificiales y generados mediante código. No contienen información personal real de ninguna persona.**

## 🎯 Objetivo del Proyecto

Este proyecto busca **generar usuarios con respuestas emocionales menos específicas y crear datos más reales** para probar modelos de machine learning. El desafío es crear datasets sintéticos que presenten la complejidad y ambigüedad de datos reales.

### 🔄 Evolución de los Datasets

Hemos desarrollado múltiples generadores de datos sintéticos para crear datasets progresivamente más desafiantes:

1. **Datos aleatorios** → 20% accuracy (nivel azar)
2. **Datos coherentes** → 95%+ accuracy (demasiado fácil, overfitting)
3. **Datos realistas** → 65-80% accuracy (objetivo actual)

## Estructura del Proyecto

```
Emotion_Analysis_Project/
├── data/
│   ├── data_emotions_coherent.csv           # Dataset coherente (fácil)
│   ├── data_clean_emotions_coherent.csv     # Procesado coherente
│   ├── data_emotions_realistic_medium.csv   # Dataset realista (desafiante)
│   └── data_clean_emotions_realistic.csv    # Procesado realista
├── generate_coherent_dataset.py             # Genera datos con patrones claros
├── generate_realistic_dataset.py            # Genera datos con ambigüedad real
├── clean_emotions_data.py                   # Procesa cualquier dataset
├── model_training.py                        # Entrenamiento con detección de overfitting
├── Inconsistencia_en_los_datosC.ipynb       # Análisis de inconsistencias - Dataset Coherente
├── Incosistencia_en_los_datosR.ipynb        # Análisis de inconsistencias - Dataset Realista
└── README.md                               # Este archivo
```

## 📊 Datasets Actuales

### 🎯 Dataset Realista (Recomendado)
**Archivo:** `data_emotions_realistic_medium.csv` → `data_clean_emotions_realistic.csv`

- **Registros:** 1,200
- **Textos únicos:** ~80% (simulando duplicados naturales)
- **Vocabulario:** 60% específico + 40% ambiguo entre emociones
- **Ruido:** 5% de etiquetas incorrectas (simulando errores humanos)
- **Accuracy esperado:** 65-80% (realista)

**Características clave:**
- Palabras ambiguas compartidas entre emociones ("malo" puede ser tristeza o enojo)
- Patrones complejos que requieren aprendizaje real
- Vocabulario superpuesto entre categorías emocionales
- Desafío similar a datos del mundo real

### 📚 Dataset Coherente (Referencia)
**Archivo:** `data_emotions_coherent.csv` → `data_clean_emotions_coherent.csv`

- **Registros:** 1,000
- **Vocabulario específico:** 100% por emoción
- **Patrones claros:** Palabras únicas para cada emoción
- **Accuracy esperado:** 95%+ (demasiado fácil)

**Uso:** Validar que el pipeline funciona correctamente antes de probar con datos realistas.

## 🧪 Justificación del Enfoque

### ❌ Problema Identificado
Los datasets sintéticos tradicionales producen **accuracy artificialmente alto (100%)** porque:
- Vocabulario demasiado específico por emoción
- Sin ambigüedad entre categorías
- Patrones demasiado obvios para los algoritmos

### ✅ Solución Implementada
**Datasets realistas** que incluyen:
- **Vocabulario ambiguo:** Palabras que pueden expresar múltiples emociones
- **Solapamiento semántico:** "terrible" puede ser tristeza, enojo o miedo
- **Ruido realista:** 5% de etiquetas incorrectas
- **Duplicados naturales:** Como ocurre en datos reales

### 🎯 Resultado Esperado
- **Accuracy realista:** 65-80% (similar a datasets reales)
- **Aprendizaje genuino:** El modelo debe encontrar patrones complejos
- **Evaluación confiable:** Métricas que reflejen rendimiento real
## 🛠️ Scripts Disponibles

### 1. Generar Dataset Coherente (Fácil)
```python
python generate_coherent_dataset.py
```
**Uso:** Validar que el pipeline funciona. Produce 95%+ accuracy.

### 2. Generar Dataset Realista (Desafiante)
```python
python generate_realistic_dataset.py
```
**Uso:** Crear datos con ambigüedad real. Objetivo de 65-80% accuracy.

### 3. Procesar y Limpiar Datos
```python
python clean_emotions_data.py
```
**Automáticamente detecta y procesa el último dataset generado.**

### 4. Entrenar y Evaluar Modelos
```python
python model_training.py
```
**Incluye detección de overfitting y análisis de rendimiento.**

### 5. Análisis de Inconsistencias - Notebooks Interactivos

#### 📊 Inconsistencia_en_los_datosC.ipynb
**Análisis exhaustivo del dataset coherente:**
- Comparación entre datos originales y procesados
- Vectorización TF-IDF y entrenamiento Random Forest
- Análisis SHAP para interpretabilidad
- Cross-validation con métricas detalladas por emoción

#### 📊 Incosistencia_en_los_datosR.ipynb  
**Análisis exhaustivo del dataset realista:**
- Evaluación de ambigüedad en vocabulario emocional
- Detección de inconsistencias en datos más complejos
- Análisis de rendimiento en condiciones realistas
- Métricas comparativas con el dataset coherente

**Características de ambos notebooks:**
- 🔍 Análisis SHAP para explicabilidad de modelos
- 📈 Cross-validation estratificado (5-folds)
- 📊 Reports detallados por clase emocional
- 🎯 Visualizaciones de dependencia de características
- 🧪 Comparación de accuracy entre datasets originales y procesados

## 📈 Estructura de Datos

**Todas las versiones comparten la misma estructura:**
- `user_id`: ID único del usuario (UUID)
- `timestamp`: Fecha y hora del registro
- `text`: Texto emocional del usuario
- `emotion`: Categoría emocional (feliz, triste, enojado, sorprendido, neutral)
- `age`: Edad del usuario (18-70 años)
- `gender`: Género (masculino, femenino, no binario)
- `region`: Región geográfica (Norte, Sur, Este, Oeste, Centro)

**Después del procesamiento:**
- Variables categóricas codificadas con One-Hot
- Texto limpio (minúsculas, sin puntuación)
- TF-IDF vectorization (calculado en tiempo real)

## 🧠 Metodología de Machine Learning

### Medidas Anti-Overfitting
- **Eliminación de duplicados:** Detecta y remueve textos idénticos
- **División estratificada:** 70% entrenamiento, 30% prueba
- **Cross-validation:** 5-fold StratifiedKFold
- **Regularización:** LogisticRegression con C=0.1
- **Detección de data leakage:** Excluye características pre-procesadas

### Modelos Evaluados
1. **Solo texto (TF-IDF):** Base para análisis de sentimientos
2. **Solo demografía:** Control de variables no textuales
3. **Texto + demografía:** Modelo combinado
4. **Random Forest:** Algoritmo alternativo

### Métricas de Evaluación
- **Accuracy:** Con contexto de baseline aleatorio (20%)
- **Cross-validation:** Para validar generalización
- **Matriz de confusión:** Errores por categoría
- **Precision/Recall/F1:** Rendimiento por emoción

## 🎯 Benchmark de Rendimiento

| Tipo de Dataset | Accuracy Esperado | Interpretación |
|------------------|-------------------|----------------|
| Aleatorio | ~20% | Nivel de azar (5 clases) |
| Coherente | 95%+ | ⚠️ Demasiado fácil, posible overfitting |
| **Realista** | **65-80%** | ✅ **Objetivo: Realista y desafiante** |

## 🔬 Investigación y Desarrollo

### 📊 Análisis de Inconsistencias de Datos

Hemos implementado un análisis exhaustivo para detectar y evaluar inconsistencias en ambos datasets utilizando técnicas avanzadas de machine learning y SHAP (SHapley Additive exPlanations).

#### 🧪 Notebooks de Análisis

1. **Inconsistencia_en_los_datosC.ipynb** - Análisis del Dataset Coherente
   - Comparación entre datos originales (`data_emotions_coherent.csv`) y procesados (`data_clean_emotions_coherent.csv`)
   - Vectorización TF-IDF con 500 características más frecuentes
   - Entrenamiento de Random Forest (n_estimators=100)
   - Análisis SHAP para interpretabilidad del modelo
   - Cross-validation con StratifiedKFold (n_splits=5)
   - Métricas detalladas por clase emocional

2. **Incosistencia_en_los_datosR.ipynb** - Análisis del Dataset Realista
   - Comparación entre datos originales (`data_emotions_realistic_medium.csv`) y procesados (`data_clean_emotions_realistic.csv`)
   - Misma metodología aplicada al dataset más desafiante
   - Análisis de rendimiento en condiciones realistas
   - Evaluación de la ambigüedad del vocabulario

#### 🔍 Metodología de Análisis

**Técnicas Implementadas:**
- **TF-IDF Vectorization:** Hasta 5000 características para análisis detallado
- **Random Forest:** Modelo ensemble para capturar patrones complejos
- **SHAP Analysis:** Explicabilidad de predicciones mediante valores Shapley
- **Cross-Validation:** Validación robusta con 5 folds estratificados
- **Classification Reports:** Métricas detalladas por clase emocional

**Métricas Evaluadas:**
- Accuracy por fold y promedio general
- Precision, Recall y F1-score por emoción
- Matriz de confusión para análisis de errores
- SHAP values para interpretación de características importantes
- Dependence plots para análisis de palabras específicas

#### 📈 Hallazgos Principales

**Dataset Coherente:**
- **Alta consistencia:** Palabras específicas claramente asociadas a emociones
- **Patrones evidentes:** Fácil separabilidad entre clases
- **Interpretabilidad clara:** SHAP values muestran características distintivas
- **Overfitting potencial:** Accuracy muy alto sugiere simplicidad excesiva

**Dataset Realista:**
- **Ambigüedad natural:** Palabras compartidas entre emociones múltiples
- **Desafío real:** Menor accuracy refleja complejidad del mundo real
- **Patrones sutiles:** SHAP revela relaciones más complejas
- **Generalización mejorada:** Mejor preparación para datos reales

#### 🛠️ Herramientas de Análisis

Las técnicas implementadas permiten:
- **Detección de inconsistencias** en etiquetado de emociones
- **Identificación de patrones** de vocabulario emocional
- **Evaluación de calidad** de datasets sintéticos
- **Optimización de modelos** mediante interpretabilidad
- **Validación cruzada robusta** para métricas confiables

### Próximos Pasos
1. **Aumentar ambigüedad:** Más solapamiento de vocabulario
2. **Contexto complejo:** Frases con emociones mixtas
3. **Ruido realista:** Simulación de errores de etiquetado humano
4. **Datos multimodales:** Incorporar metadata temporal/demográfica

### Casos de Uso
- **Investigación:** Evaluación realista de modelos de NLP
- **Educación:** Enseñanza de machine learning con datos challenging
- **Desarrollo:** Testing de algoritmos de análisis de sentimientos
- **Benchmarking:** Comparación de técnicas de procesamiento de texto

## Requisitos

```bash
# Dependencias básicas del proyecto
pip install pandas numpy faker scikit-learn scipy

# Dependencias adicionales para análisis de inconsistencias
pip install matplotlib seaborn shap jupyter

# Instalación completa recomendada
pip install pandas numpy faker scikit-learn scipy matplotlib seaborn shap jupyter
```

**Dependencias por funcionalidad:**
- **Generación de datos:** `pandas`, `numpy`, `faker`
- **Machine learning:** `scikit-learn`, `scipy`
- **Análisis de inconsistencias:** `matplotlib`, `seaborn`, `shap`
- **Notebooks interactivos:** `jupyter`

## 📄 Licencia

**DATOS SINTÉTICOS - USO LIBRE**

Este proyecto utiliza datos completamente artificiales. Libre para uso educativo, investigación y comercial sin restricciones.

---

## 🎯 **Conclusiones: Impacto de las Inconsistencias en la Efectividad del Modelo**

### 📊 **Análisis Comparativo de Coherencia de Datos**

Nuestro análisis exhaustivo mediante los notebooks de inconsistencias revela patrones críticos que afectan directamente la efectividad de los modelos de machine learning:

#### 🔴 **Factores que Reducen la Efectividad del Modelo**

**1. Ambigüedad Semántica**
- **Vocabulario superpuesto:** Palabras como "malo", "terrible" aparecen en múltiples emociones (tristeza, enojo, miedo)
- **Impacto:** El modelo no puede establecer relaciones claras entre características y etiquetas
- **Evidencia SHAP:** Los valores de importancia se distribuyen inconsistentemente entre palabras ambiguas

**2. Ruido en Etiquetado**
- **5% de etiquetas incorrectas** en el dataset realista simula errores humanos reales
- **Impacto:** Confunde el algoritmo durante el entrenamiento, reduciendo la confianza en predicciones
- **Resultado:** Accuracy baja de 95% (coherente) a 65-80% (realista)

**3. Inconsistencia en Patrones Textuales**
- **Dataset coherente:** Cada emoción tiene vocabulario único y específico
- **Dataset realista:** Múltiples emociones comparten el mismo vocabulario base
- **Consecuencia:** El modelo debe aprender relaciones más sutiles y contextuales

#### ✅ **Validación de Hipótesis**

**Comparación de Rendimiento:**
| Aspecto | Dataset Coherente | Dataset Realista | Impacto en Efectividad |
|---------|-------------------|------------------|------------------------|
| **Accuracy** | 95%+ | 65-80% | ⬇️ **-15 a -30%** |
| **Separabilidad** | Clara | Ambigua | ⬇️ **Decisiones inciertas** |
| **Interpretabilidad SHAP** | Específica | Distribuida | ⬇️ **Explicaciones confusas** |
| **Generalización** | Overfitting | Robusta | ✅ **Mejor en datos reales** |

#### 🧠 **Mecanismos de Impacto Identificados**

**1. Degradación de la Función de Pérdida**
- Las inconsistencias crean **contradicciones** en los datos de entrenamiento
- El modelo no puede minimizar eficientemente la pérdida
- **Resultado:** Convergencia lenta y rendimiento subóptimo

**2. Reducción de la Capacidad Predictiva**
- **Características ruidosas** dominan sobre señales genuinas
- El modelo aprende patrones espurios en lugar de relaciones reales
- **Consecuencia:** Predicciones menos confiables

**3. Complejidad de Decisión Aumentada**
- **Fronteras de decisión difusas** entre clases emocionales
- Requiere algoritmos más sofisticados para capturar sutilezas
- **Trade-off:** Mayor realismo vs menor accuracy inmediata

#### 🎯 **Implicaciones para el Mundo Real**

**Lecciones Aprendidas:**

1. **La coherencia perfecta es irreal:** Los datos del mundo real siempre contienen ambigüedades
2. **El overfitting es peligroso:** Un accuracy del 95% en datos sintéticos puede ser engañoso
3. **La ambigüedad es valiosa:** Datasets realistas preparan mejor los modelos para casos reales
4. **La interpretabilidad sufre:** SHAP muestra patrones más complejos en datos inconsistentes

**Recomendaciones Estratégicas:**

- ✅ **Usar datasets realistas** para evaluación final de modelos
- ✅ **Validar con cross-validation** para detectar overfitting
- ✅ **Analizar SHAP values** para entender decisiones del modelo
- ✅ **Aceptar accuracy menor** si refleja condiciones reales
- ⚠️ **Desconfiar de accuracy > 90%** en datos emocionales complejos

#### 📈 **Valor del Análisis de Inconsistencias**

Este proyecto demuestra que **la inconsistencia controlada en datos sintéticos** es fundamental para:
- **Evaluación realista** de algoritmos de NLP
- **Preparación robusta** de modelos para producción
- **Comprensión profunda** de limitaciones algorítmicas
- **Desarrollo responsable** de IA emocional

**Conclusión Final:** Las inconsistencias en los datos, aunque reducen métricas superficiales como el accuracy, **mejoran significativamente** la capacidad del modelo para generalizar a datos reales, proporcionando una evaluación más honesta y útil del rendimiento algorítmico.