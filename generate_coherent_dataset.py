import pandas as pd
import numpy as np
import random
from faker import Faker
import re

fake = Faker()

# Vocabulario específico por emoción
EMOTION_VOCABULARIES = {
    'feliz': {
        'words': ['happy', 'joy', 'excited', 'wonderful', 'amazing', 'great', 'fantastic', 'love', 'smile', 'celebrate', 
                 'awesome', 'perfect', 'brilliant', 'delighted', 'cheerful', 'pleased', 'thrilled', 'ecstatic'],
        'phrases': ['I am so happy', 'This is wonderful', 'I love this', 'What a great day', 'I feel amazing',
                   'This makes me smile', 'I am thrilled', 'How fantastic', 'I feel incredible'],
        'endings': ['!', '! :)', '! 😊', '! This is great!', '! I love it!']
    },
    'triste': {
        'words': ['sad', 'upset', 'disappointed', 'hurt', 'crying', 'lonely', 'depressed', 'miserable', 'down', 'heartbroken',
                 'terrible', 'awful', 'bad', 'unfortunate', 'devastating', 'painful', 'gloomy'],
        'phrases': ['I am so sad', 'This is terrible', 'I feel awful', 'This hurts', 'I am disappointed',
                   'This makes me cry', 'I feel down', 'This is devastating', 'I am heartbroken'],
        'endings': ['...', '. :(', '. I feel terrible', '. This is awful', '. Very disappointing']
    },
    'enojado': {
        'words': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated', 'outraged', 'livid', 'upset', 'pissed',
                 'hate', 'stupid', 'ridiculous', 'unacceptable', 'disgusting', 'infuriating'],
        'phrases': ['I am so angry', 'This is ridiculous', 'I hate this', 'This is unacceptable', 'I am furious',
                   'This makes me mad', 'I am frustrated', 'This is stupid', 'I am outraged'],
        'endings': ['!', '! This is ridiculous!', '! I hate this!', '! Unacceptable!', '! So frustrated!']
    },
    'sorprendido': {
        'words': ['surprised', 'shocked', 'amazed', 'unexpected', 'incredible', 'unbelievable', 'wow', 'astonished', 'stunned',
                 'remarkable', 'extraordinary', 'mind-blowing', 'fascinating'],
        'phrases': ['I am so surprised', 'This is incredible', 'I cannot believe this', 'What a shock', 'This is amazing',
                   'I am stunned', 'This is unexpected', 'Wow this is', 'I am astonished'],
        'endings': ['!', '! Wow!', '! I cannot believe it!', '! What a shock!', '! Incredible!']
    },
    'neutral': {
        'words': ['okay', 'fine', 'normal', 'regular', 'standard', 'usual', 'typical', 'average', 'moderate', 'reasonable',
                 'acceptable', 'adequate', 'sufficient', 'common'],
        'phrases': ['This is okay', 'It is fine', 'Nothing special', 'This is normal', 'It is adequate',
                   'This is typical', 'It is average', 'This is standard', 'It is acceptable'],
        'endings': ['.', '. Nothing special.', '. It is okay.', '. Pretty normal.', '. Just fine.']
    }
}

def generate_coherent_text(emotion, length='medium'):
    """Genera texto coherente para una emoción específica"""
    vocab = EMOTION_VOCABULARIES[emotion]
    
    # Agregar variabilidad para evitar duplicados
    if length == 'short':
        # Combinar frase base con variaciones
        base = random.choice(vocab['phrases'])
        variation = random.choice(['', f" {random.choice(vocab['words'])}", f" so {random.choice(vocab['words'])}"])
        ending = random.choice(vocab['endings'])
        return base + variation + ending
    
    elif length == 'medium':
        # Más variabilidad en construcción
        templates = [
            "{phrase} because it is {words}",
            "{phrase}. Everything feels {words}",
            "{phrase} and I think it's {words}",
            "I believe {phrase} since it's {words}",
            "Today {phrase} because things are {words}"
        ]
        template = random.choice(templates)
        phrase = random.choice(vocab['phrases'])
        context_words = random.sample(vocab['words'], min(3, len(vocab['words'])))
        words_str = ', '.join(context_words[:-1]) + f' and {context_words[-1]}'
        ending = random.choice(vocab['endings'])
        return template.format(phrase=phrase, words=words_str) + ending
    
    else:  # long
        # Estructura más compleja
        templates = [
            "{phrase1}. {phrase2}. I really think {context}",
            "Honestly, {phrase1}. {phrase2} and {context}",
            "Today {phrase1}. Plus, {phrase2} because {context}",
            "{phrase1}! {phrase2} and everything feels {context}"
        ]
        template = random.choice(templates)
        phrase1 = random.choice(vocab['phrases'])
        phrase2 = f"Everything feels {random.choice(vocab['words'])}"
        context_words = random.sample(vocab['words'], min(4, len(vocab['words'])))
        context = f"this is {context_words[0]} and {context_words[1]}"
        ending = random.choice(vocab['endings'])
        return template.format(phrase1=phrase1, phrase2=phrase2, context=context) + ending

def generate_realistic_emotion_dataset(n_samples=1000):
    """Genera dataset con correlación texto-emoción"""
    
    data = []
    emotions = list(EMOTION_VOCABULARIES.keys())
    generos = ["masculino", "femenino", "no binario"]
    regiones = ["Norte", "Sur", "Este", "Oeste", "Centro"]
    
    for i in range(n_samples):
        # Seleccionar emoción
        emotion = random.choice(emotions)
        
        # Generar texto coherente con la emoción
        text_length = random.choice(['short', 'medium', 'long'])
        text = generate_coherent_text(emotion, text_length)
        
        # Otros datos
        user_id = fake.uuid4()
        timestamp = fake.date_time_between(start_date='-2y', end_date='now')
        age = np.random.randint(18, 70)
        gender = random.choice(generos)
        region = random.choice(regiones)
        
        data.append([user_id, timestamp, text, emotion, age, gender, region])
    
    # Crear DataFrame
    df = pd.DataFrame(data, columns=[
        "user_id", "timestamp", "text", "emotion", "age", "gender", "region"
    ])
    
    return df

def generate_mixed_dataset(n_realistic=800, n_random=200):
    """Combina datos realistas con algunos aleatorios para simular ruido real"""
    
    # Datos realistas
    df_realistic = generate_realistic_emotion_dataset(n_realistic)
    
    # Algunos datos con ruido (como en datos reales)
    data_noise = []
    emotions = list(EMOTION_VOCABULARIES.keys())
    
    for i in range(n_random):
        emotion = random.choice(emotions)
        # Texto completamente aleatorio (simula ruido en datos reales)
        text = fake.sentence(nb_words=random.randint(5, 15))
        
        user_id = fake.uuid4()
        timestamp = fake.date_time_between(start_date='-2y', end_date='now')
        age = np.random.randint(18, 70)
        gender = random.choice(["masculino", "femenino", "no binario"])
        region = random.choice(["Norte", "Sur", "Este", "Oeste", "Centro"])
        
        data_noise.append([user_id, timestamp, text, emotion, age, gender, region])
    
    df_noise = pd.DataFrame(data_noise, columns=[
        "user_id", "timestamp", "text", "emotion", "age", "gender", "region"
    ])
    
    # Combinar y mezclar
    df_combined = pd.concat([df_realistic, df_noise], ignore_index=True)
    df_combined = df_combined.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    return df_combined

if __name__ == "__main__":
    print("Generando dataset con correlación texto-emoción...")
    
    # Opción 1: Dataset completamente coherente
    print("\n=== GENERANDO DATASET COHERENTE ===")
    df_coherent = generate_realistic_emotion_dataset(1000)
    
    # Guardar
    df_coherent.to_csv('data/data_emotions_coherent.csv', index=False)
    print(f"✅ Dataset coherente guardado: data/data_emotions_coherent.csv")
    print(f"   Forma: {df_coherent.shape}")
    
    # Mostrar ejemplos
    print(f"\n=== EJEMPLOS DE TEXTO COHERENTE ===")
    for emotion in df_coherent['emotion'].unique():
        sample = df_coherent[df_coherent['emotion'] == emotion].iloc[0]
        print(f"{emotion:12}: {sample['text']}")
    
    # Opción 2: Dataset mixto (80% coherente, 20% ruido)
    print(f"\n=== GENERANDO DATASET MIXTO ===")
    df_mixed = generate_mixed_dataset(800, 200)
    
    # Guardar
    df_mixed.to_csv('data/data_emotions_mixed.csv', index=False)
    print(f"✅ Dataset mixto guardado: data/data_emotions_mixed.csv")
    print(f"   Forma: {df_mixed.shape}")
    print(f"   80% coherente + 20% ruido (simula datos reales)")
    
    print(f"\n=== DISTRIBUCIÓN DE EMOCIONES ===")
    print(df_coherent['emotion'].value_counts())
    
    print(f"\n=== PRÓXIMOS PASOS ===")
    print("1. 🧹 Ejecuta clean_emotions_data.py con el nuevo dataset")
    print("2. 🤖 Entrena modelos y compara accuracy")
    print("3. 📊 Deberías ver accuracy >70% con datos coherentes")
    print("4. 🔬 Usa el dataset mixto para simular condiciones reales")