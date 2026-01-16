import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# MANIPULACIÓN Y LIMPIEZA DE DATOS FICTICIOS

# Cargar el dataset
df = pd.read_csv('data/data_emociones_faker.csv')

# Revisar las primeras filas
print(df.head())

# Revisar los tipos de datos
print(df.dtypes)

# Revisar info general
print(df.info())

# Revisar valores nulos
print(df.isnull().sum())

# Revisar duplicados
print(df.duplicated().sum())

# Revisar estadísticas descriptivas
print(df.describe())

# Eliminar duplicados
df = df.drop_duplicates()

# Convertir la columna timestamp a tipo datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Filtrar edades válidas(opcional)
df = df[(df['age'] >= 0) & (df['age'] <= 120)]

# Definir columnas categóricas
categorical_cols = ["gender", "region"]

# Codificación One_Hot (elimina la primera categoría para evitar multicolinealidad)
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_cats = encoder.fit_transform(df[categorical_cols])

# Crear DataFrame con columnas codificadas
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

# Combinar con DataFrame original y eliminar columnas originales
df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)

# Función para limpiar texto
def clean_text(text):
    text = text.lower()                             # pasar a minúsculas
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)    # eliminar puntuación y caracteres especiales
    text = re.sub(r'\s+', ' ', text).strip()       # eliminar espacios extra
    return text

# Aplicar limpieza
df["text_clean"] = df["text"].apply(clean_text)

# Revisar resultados
print(df[["text", "text_clean"]].head())

# Escalar variables numéricas
num_cols = ['age']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Guardar CSV limpio
df.to_csv('data/data_clean_emotions.csv', index=False)

print("Dataset limpio guardado como: data_clean_emotions.csv")
print(f"Shape final: {df.shape}")
print("\nDistribución de emociones:")
print(df['emotion'].value_counts())