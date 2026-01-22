import pandas as pd
import numpy as np
import random
from faker import Faker
import re

fake = Faker()

# Vocabulario más realista con solapamiento y ambigüedad
EMOTION_VOCABULARIES_REALISTIC = {
    'feliz': {
        'primary': ['happy', 'joy', 'great', 'wonderful', 'love', 'smile', 'excited'],
        'secondary': ['good', 'nice', 'okay', 'fine', 'well', 'positive', 'bright'],  # Palabras ambiguas
        'phrases': ['I feel good', 'This is nice', 'I am okay with this', 'It went well', 'Pretty good'],
        'endings': ['.', '!', '. Good.', '. Nice.']
    },
    'triste': {
        'primary': ['sad', 'terrible', 'awful', 'devastating', 'hurt', 'crying'],
        'secondary': ['bad', 'not good', 'difficult', 'hard', 'tough', 'disappointing'],  # Ambiguas
        'phrases': ['I feel bad', 'This is not good', 'It is difficult', 'This is hard', 'Pretty bad'],
        'endings': ['.', '...', '. Bad.', '. Not good.']
    },
    'enojado': {
        'primary': ['angry', 'mad', 'furious', 'outraged', 'hate', 'ridiculous'],
        'secondary': ['bad', 'not good', 'stupid', 'wrong', 'terrible', 'annoying'],  # Solapamiento
        'phrases': ['This is wrong', 'This is bad', 'I do not like this', 'This is annoying', 'Not good'],
        'endings': ['!', '.', '! Wrong!', '. Bad.']
    },
    'sorprendido': {
        'primary': ['surprised', 'shocked', 'wow', 'incredible', 'unbelievable', 'amazing'],
        'secondary': ['interesting', 'unexpected', 'different', 'strange', 'unusual'],  # Neutrales
        'phrases': ['This is interesting', 'How unexpected', 'This is different', 'Pretty unusual', 'Quite strange'],
        'endings': ['!', '.', '! Wow!', '. Interesting.']
    },
    'neutral': {
        'primary': ['okay', 'fine', 'normal', 'regular', 'standard', 'typical'],
        'secondary': ['good', 'bad', 'nice', 'interesting', 'different'],  # Muy ambiguas
        'phrases': ['It is okay', 'This is fine', 'Seems normal', 'Pretty standard', 'It is what it is'],
        'endings': ['.', '. Fine.', '. Okay.', '. Normal.']
    }
}

def generate_realistic_text(emotion, complexity='medium'):
    """Genera texto más realista con ambigüedad y solapamiento"""
    vocab = EMOTION_VOCABULARIES_REALISTIC[emotion]
    
    # Probabilidades de usar palabras primarias vs secundarias
    if complexity == 'easy':
        use_primary_prob = 0.8  # 80% palabras específicas
    elif complexity == 'medium':
        use_primary_prob = 0.6  # 60% palabras específicas
    else:  # hard
        use_primary_prob = 0.4  # 40% palabras específicas (más ambiguas)
    
    # Seleccionar tipo de palabras
    if random.random() < use_primary_prob:
        words = vocab['primary']
    else:
        words = vocab['secondary']  # Palabras ambiguas
    
    # Construir texto
    if random.random() < 0.3:  # 30% frases predefinidas
        text = random.choice(vocab['phrases'])
    else:  # 70% construcción dinámica
        templates = [
            "I feel {word}",
            "This is {word}",
            "It seems {word}",
            "Everything is {word}",
            "I think it is {word}",
            "This makes me feel {word}",
            "It looks {word}",
            "Seems quite {word}"
        ]
        template = random.choice(templates)
        word = random.choice(words)
        text = template.format(word=word)
    
    # Agregar variabilidad extra
    if random.random() < 0.3:  # 30% agregar contexto
        extra_word = random.choice(words)
        text += f" and {extra_word}"
    
    ending = random.choice(vocab['endings'])
    return text + ending

def generate_challenging_dataset(n_samples=1000, difficulty='medium'):
    """Genera dataset más desafiante para ML"""
    
    data = []
    emotions = list(EMOTION_VOCABULARIES_REALISTIC.keys())
    generos = ["masculino", "femenino", "no binario"]
    regiones = ["Norte", "Sur", "Este", "Oeste", "Centro"]
    
    # Distribución no perfectamente balanceada (más realista)
    emotion_weights = [0.22, 0.18, 0.25, 0.20, 0.15]  # Ligeramente desbalanceado
    
    print(f"Generando dataset desafiante (dificultad: {difficulty})...")
    
    for i in range(n_samples):
        # Seleccionar emoción con distribución realista
        emotion = np.random.choice(emotions, p=emotion_weights)
        
        # Generar texto más realista
        text = generate_realistic_text(emotion, difficulty)
        
        # Agregar 5% de etiquetas ruidosas (más realista)
        if random.random() < 0.05:
            emotion = random.choice([e for e in emotions if e != emotion])
        
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

if __name__ == "__main__":
    print("Generando dataset REALISTA para ML...")
    
    # Generar diferentes niveles de dificultad
    difficulties = ['easy', 'medium', 'hard']
    
    for difficulty in ['medium']:  # Usar solo medium por ahora
        print(f"\n=== GENERANDO DATASET {difficulty.upper()} ===")
        
        df = generate_challenging_dataset(1200, difficulty)  # Más samples
        
        # Verificar diversidad de texto
        unique_texts = df['text'].nunique()
        total_texts = len(df)
        diversity = unique_texts / total_texts
        
        print(f"✅ Dataset {difficulty} generado:")
        print(f"   Total de registros: {total_texts}")
        print(f"   Textos únicos: {unique_texts} ({diversity:.1%})")
        print(f"   Diversidad: {'Alta' if diversity > 0.9 else 'Media' if diversity > 0.7 else 'Baja'}")
        
        # Guardar
        filename = f'data/data_emotions_realistic_{difficulty}.csv'
        df.to_csv(filename, index=False)
        print(f"   Guardado: {filename}")
        
        # Mostrar ejemplos
        print(f"\n=== EJEMPLOS {difficulty.upper()} ===")
        for emotion in df['emotion'].unique():
            sample = df[df['emotion'] == emotion].iloc[0]
            print(f"{emotion:12}: {sample['text']}")
        
        # Distribución
        print(f"\n=== DISTRIBUCIÓN {difficulty.upper()} ===")
        dist = df['emotion'].value_counts()
        for emotion, count in dist.items():
            print(f"{emotion:12}: {count} ({count/total_texts:.1%})")
    
    print(f"\n=== COMPARACIÓN CON DATASET ANTERIOR ===")
    print("ANTERIOR (demasiado fácil):")
    print("  - Vocabulario 100% específico por emoción")
    print("  - Sin ambigüedad ni solapamiento")
    print("  - Accuracy artificial del 100%")
    print()
    print("NUEVO (más realista):")
    print("  - 60% palabras específicas, 40% ambiguas")
    print("  - Solapamiento entre emociones")
    print("  - 5% etiquetas ruidosas")
    print("  - Accuracy esperado: 65-80% (más realista)")
    
    print(f"\n=== PRÓXIMOS PASOS ===")
    print("1. 🧹 Ejecuta: python clean_emotions_data.py")
    print("   (Actualizar para usar el nuevo archivo)")
    print("2. 🤖 Ejecuta: python model_training.py")
    print("3. 📊 Accuracy esperado: 65-80% (mucho más realista)")
    print("4. 🔬 Los modelos tendrán que trabajar más para aprender")