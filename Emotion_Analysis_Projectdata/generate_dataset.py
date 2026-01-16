import pandas as pd
import numpy as np
from faker import Faker
import random

# Inicializar Faker
fake = Faker()

# Número de registros que queremos generar
n = 500  # Puedes cambiarlo según lo que necesites

# Categorías posibles
emociones = ["feliz", "triste", "enojado", "sorprendido", "neutral"]
generos = ["masculino", "femenino", "no binario"]
regiones = ["Norte", "Sur", "Este", "Oeste", "Centro"]

# Lista para guardar los datos generados
data = []

# Generación de datos
for i in range(n):
    # ID único del usuario
    user_id = fake.uuid4()
    
    # Fecha y hora aleatoria dentro de los últimos 2 años
    timestamp = fake.date_time_between(start_date='-2y', end_date='now')
    
    # Texto aleatorio (comentario o frase del usuario)
    text = fake.sentence(nb_words=random.randint(5, 15))
    
    # Emoción aleatoria
    emotion = random.choice(emociones)
    
    # Edad del usuario
    age = np.random.randint(18, 70)
    
    # Género aleatorio
    gender = random.choice(generos)
    
    # Región aleatoria
    region = random.choice(regiones)
    
    # Agregar fila al dataset
    data.append([user_id, timestamp, text, emotion, age, gender, region])

# Crear DataFrame con pandas
df_emotions = pd.DataFrame(data, columns=[
    "user_id", "timestamp", "text", "emotion", "age", "gender", "region"
])

# Guardar el dataset como CSV en la ruta absoluta
import os
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "data_emociones_faker.csv")
df_emotions.to_csv(output_path, index=False)
print(f"Dataset generado: {output_path}")
