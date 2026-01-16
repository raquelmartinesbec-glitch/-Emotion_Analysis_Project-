import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# MANIPULACIÓN Y LIMPIEZA DE DATOS FICTICIOS

def clean_text(text):
    """Función para limpiar texto"""
    text = text.lower()                             # pasar a minúsculas
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)    # eliminar puntuación y caracteres especiales
    text = re.sub(r'\s+', ' ', text).strip()       # eliminar espacios extra
    return text

def clean_emotions_data(file_path='data/data_emociones_faker.csv'):
    """
    Función principal para limpiar y procesar el dataset de emociones
    """
    print("=== CARGANDO DATASET ===")
    #Cargar el dataset
    df = pd.read_csv(file_path)
    print(f"Dataset cargado: {len(df)} registros")
    
    print("\n=== INFORMACIÓN INICIAL ===")
    #Revisar las primeras filas
    print("Primeras 5 filas:")
    print(df.head())
    
    #Revisar los tipos de datos
    print("\nTipos de datos:")
    print(df.dtypes)
    
    #Revisar info general
    print("\nInformación general:")
    print(df.info())
    
    #Revisar valores nulos
    print("\nValores nulos:")
    print(df.isnull().sum())
    
    #Revisar duplicados
    print(f"\nDuplicados encontrados: {df.duplicated().sum()}")
    
    #Revisar estadísticas descriptivas
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    print("\n=== LIMPIEZA DE DATOS ===")
    #Eliminar duplicados
    original_rows = len(df)
    df = df.drop_duplicates()
    print(f"Duplicados eliminados: {original_rows - len(df)}")
    
    #Convertir la columna timestamp a tipo datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print("Timestamp convertido a datetime")
    
    #Filtrar edades válidas (opcional)
    before_filter = len(df)
    df = df[(df['age'] >= 0) & (df['age'] <= 120)]
    print(f"Registros con edades inválidas eliminados: {before_filter - len(df)}")
    
    print("\n=== CODIFICACIÓN DE VARIABLES CATEGÓRICAS ===")
    #Definir columnas categóricas
    categorical_cols = ["gender", "region"]
    
    #Codificación One_Hot (elimina la primera categoría para evitar multicolinealidad)
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_cats = encoder.fit_transform(df[categorical_cols])
    
    # Crear DataFrame con columnas codificadas
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Combinar con DataFrame original y eliminar columnas originales
    df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
    print(f"Variables categóricas codificadas: {list(encoder.get_feature_names_out(categorical_cols))}")
    
    print("\n=== LIMPIEZA DE TEXTO ===")
    # Crear columna de texto limpio
    df["text_clean"] = df["text"].apply(clean_text)
    
    #Revisar resultados
    print("Ejemplo de limpieza de texto:")
    print(df[["text", "text_clean"]].head())
    
    print("\n=== ESCALAMIENTO DE VARIABLES NUMÉRICAS ===")
    #Escalar variables numéricas
    num_cols = ['age']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"Variables escaladas: {num_cols}")
    
    print("\n=== GUARDANDO DATOS LIMPIOS ===")
    # Guardar CSV limpio
    output_path = 'data/data_clean_emotions.csv'
    df.to_csv(output_path, index=False)
    print(f"Datos limpios guardados en: {output_path}")
    
    print("\n=== RESUMEN FINAL ===")
    print(f"Registros finales: {len(df)}")
    print(f"Columnas finales: {len(df.columns)}")
    print("Columnas disponibles:")
    for col in df.columns:
        print(f"  - {col}")
    
    return df, encoder, scaler

# Ejecutar limpieza
if __name__ == "__main__":
    df_clean, encoder, scaler = clean_emotions_data()
    
    # Mostrar estadísticas finales
    print("\n=== DATOS LISTOS PARA MACHINE LEARNING ===")
    print(f"Shape final: {df_clean.shape}")
    print("\nDistribución de emociones:")
    print(df_clean['emotion'].value_counts())