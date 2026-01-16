# Emotion Analysis Project

Proyecto de análisis de emociones utilizando **datos completamente sintéticos y ficticios** generados con la librería Faker.

⚠️ **IMPORTANTE: Todos los datos en este proyecto son completamente artificiales y generados mediante código. No contienen información personal real de ninguna persona. Los datos pueden ser utilizados libremente para fines educativos, de investigación y comerciales.**

## Estructura del Proyecto

```
Emotion_Analysis_Project/
├── data/
│   ├── data_emociones_faker.csv      # Dataset original generado
│   └── data_clean_emotions.csv       # Dataset limpio y procesado
├── generate_dataset.py               # Script para generar datos sintéticos
├── clean_emotions_data.py             # Script para limpiar y procesar datos
├── data_analysis.py                   # Script básico para análisis
└── README.md                         # Este archivo
```

## Datasets Disponibles

### 1. Dataset Original (data_emociones_faker.csv)
- **Registros:** 500
- **Columnas:** 7
- **Descripción:** Datos completamente sintéticos y ficticios de usuarios con emociones generados usando Faker
- **⚠️ DATOS FICTICIOS:** Todos los valores son artificiales, no corresponden a personas reales

**Estructura:**
- `user_id`: ID único del usuario (UUID)
- `timestamp`: Fecha y hora del registro
- `text`: Texto de ejemplo/comentario del usuario
- `emotion`: Emoción etiquetada (feliz, triste, enojado, sorprendido, neutral)
- `age`: Edad del usuario (18-70 años)
- `gender`: Género (masculino, femenino, no binario)
- `region`: Región geográfica (Norte, Sur, Este, Oeste, Centro)

### 2. Dataset Limpio (data_clean_emotions.csv)
- **Registros:** 501 (después de procesamiento)
- **Columnas:** 11 (después de codificación)
- **Descripción:** Datos sintéticos procesados y listos para machine learning
- **⚠️ DATOS FICTICIOS:** Todos los valores son artificiales, no corresponden a personas reales

**Procesamiento aplicado:**
- Eliminación de duplicados
- Conversión de timestamp a datetime
- Filtrado de edades válidas (0-120)
- Codificación One-Hot de variables categóricas:
  - `gender_masculino`, `gender_no binario` (género femenino como referencia)
  - `region_Este`, `region_Norte`, `region_Oeste`, `region_Sur` (región Centro como referencia)
- Limpieza de texto (minúsculas, sin puntuación)
- Escalamiento de edad usando StandardScaler

**Columnas finales:**
- `user_id`: ID único del usuario
- `timestamp`: Fecha y hora (datetime)
- `text`: Texto original
- `emotion`: Emoción objetivo (target variable)
- `age`: Edad escalada (StandardScaler)
- `gender_masculino`: Variable binaria (1 = masculino, 0 = no)
- `gender_no binario`: Variable binaria (1 = no binario, 0 = no)
- `region_Este`: Variable binaria (1 = Este, 0 = no)
- `region_Norte`: Variable binaria (1 = Norte, 0 = no)
- `region_Oeste`: Variable binaria (1 = Oeste, 0 = no)
- `region_Sur`: Variable binaria (1 = Sur, 0 = no)

## Uso de los Scripts

### Generar datos sintéticos
```python
python generate_dataset.py
```

### Limpiar y procesar datos
```python
python clean_emotions_data.py
```

### Análisis básico
```python
python data_analysis.py
```

## Distribución de Emociones

El dataset tiene una distribución aproximadamente balanceada entre las 5 categorías de emociones:
- feliz
- triste
- enojado
- sorprendido
- neutral

## Acceso a los Datos

**🔓 DATOS PÚBLICOS Y LIBRES:** Todos los datos en este proyecto son completamente sintéticos y ficticios, generados mediante código. No contienen información personal real.

Los archivos CSV están disponibles en la carpeta `data/` y pueden ser utilizados libremente para:
- Análisis exploratorio de datos
- Entrenamiento de modelos de machine learning
- Análisis de sentimientos
- Práctica con procesamiento de lenguaje natural
- Proyectos educativos
- Uso comercial

**No se requiere permiso especial ya que todos los datos son artificiales.**

## Requisitos

```bash
pip install pandas numpy faker scikit-learn matplotlib seaborn
```

## Licencia

**DATOS SINTÉTICOS - USO LIBRE**

Este proyecto y los datos sintéticos están disponibles para uso libre en cualquier contexto (educativo, comercial, investigación) ya que todos los datos son completamente artificiales y generados por código. No contienen información personal real de ninguna persona.

Los datos pueden ser redistribuidos, modificados y utilizados sin restricciones.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request para mejoras.