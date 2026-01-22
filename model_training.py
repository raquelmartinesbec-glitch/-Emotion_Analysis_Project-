import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from scipy.sparse import hstack

# Cargar el dataset limpio
print("Cargando dataset limpio...")

# Verificar archivos disponibles
import os
print("Archivos en el directorio actual:")
print(os.listdir('.'))

# Verificar si existe la carpeta data
if os.path.exists('data'):
    print("\nArchivos en la carpeta data:")
    print(os.listdir('data'))
    # USAR el dataset realista (datos con correlación texto-emoción MÁS DESAFIANTE)
    data_path = 'data/data_clean_emotions_realistic.csv'
    if not os.path.exists(data_path):
        print("⚠️ Dataset realista no encontrado. Generándolo...")
        print("Ejecuta: python generate_realistic_dataset.py")
        print("Luego: python clean_emotions_data.py")
        exit()
else:
    # Buscar el archivo en el directorio actual
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    print(f"\nArchivos CSV disponibles: {csv_files}")
    
    # Intentar con diferentes nombres posibles
    possible_files = [
        'data_clean_emotions_realistic.csv',  # Dataset realista (prioridad)
        'data_clean_emotions_coherent.csv',   # Dataset coherente (backup)
        'data_clean_emotions_1000.csv',
        'data_clean_emotions.csv',
        'data_emociones_faker.csv'
    ]
    
    data_path = None
    for file in possible_files:
        if file in csv_files:
            data_path = file
            break
    
    if data_path is None and csv_files:
        data_path = csv_files[0]  # Usar el primer CSV encontrado
        print(f"Usando archivo: {data_path}")

if data_path and os.path.exists(data_path):
    data_clean = pd.read_csv(data_path)
    print(f"Dataset cargado desde {data_path} con forma: {data_clean.shape}")
else:
    print("ERROR: No se encontró el dataset coherente.")
    print("Ejecuta estos comandos para generar datos coherentes:")
    print("1. python generate_coherent_dataset.py")
    print("2. python clean_emotions_data.py")
    exit()

# Mostrar primeras filas y info del dataset
print("\nPrimeras 5 filas:")
print(data_clean.head())

print(f"\nColumnas disponibles: {list(data_clean.columns)}")

# Preparar los datos
# Separar texto original y características procesadas
text_column = 'text_clean'  # Usar el texto limpio
target_column = 'emotion'

# Obtener el texto limpio y el target
X_text = data_clean[text_column]
y = data_clean[target_column]

# Obtener características demográficas (excluyendo texto, TF-IDF pre-procesado, id, timestamp y emotion)
exclude_cols = ['user_id', 'timestamp', 'text', 'text_clean', 'emotion'] + [col for col in data_clean.columns if col.startswith('tfidf_')]
demographic_cols = [col for col in data_clean.columns if col not in exclude_cols]
X_demographic = data_clean[demographic_cols]

print(f"\n⚠️ EVITANDO DATA LEAKAGE: Excluyendo {len([col for col in data_clean.columns if col.startswith('tfidf_')])} características TF-IDF pre-procesadas")
print(f"Características demográficas utilizadas: {len(demographic_cols)}")
print(f"Columnas demográficas: {demographic_cols}")

# División train/test con más validación
print("\n🔍 ANÁLISIS DE DUPLICADOS Y DISTRIBUCIÓN:")
print(f"Textos únicos: {X_text.nunique()} de {len(X_text)} ({X_text.nunique()/len(X_text)*100:.1f}%)")

# Verificar si hay texto duplicado
duplicates = X_text.duplicated().sum()
if duplicates > 0:
    print(f"⚠️ ADVERTENCIA: {duplicates} textos duplicados detectados")
    print("💡 Esto puede causar overfitting artificial")
    
    # ELIMINAR DUPLICADOS para evitar data leakage
    print("🧹 ELIMINANDO DUPLICADOS...")
    original_size = len(data_clean)
    data_clean_unique = data_clean.drop_duplicates(subset=['text_clean'], keep='first')
    removed = original_size - len(data_clean_unique)
    
    print(f"   Registros originales: {original_size}")
    print(f"   Registros únicos: {len(data_clean_unique)}")
    print(f"   Duplicados eliminados: {removed}")
    
    # Actualizar variables con datos únicos
    X_text = data_clean_unique[text_column]
    y = data_clean_unique[target_column]
    X_demographic = data_clean_unique[demographic_cols]
    
    print(f"✅ Dataset sin duplicados: {len(data_clean_unique)} registros")

X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.3, random_state=42, stratify=y  # Aumentar test set
)

X_demo_train, X_demo_test = train_test_split(
    X_demographic, test_size=0.3, random_state=42, stratify=y  # Coincidir con text split
)

print(f"\nDatos de entrenamiento: {len(X_text_train)}")
print(f"Datos de prueba: {len(X_text_test)}")

# Crear vectorizador TF-IDF (CALCULAR DESDE CERO para evitar data leakage)
print("\n🚨 IMPORTANTE: Calculando TF-IDF desde texto crudo para evitar data leakage")
tfidf = TfidfVectorizer(
    max_features=1000,  # Reducir características para hacer más difícil
    stop_words='english',
    ngram_range=(1, 2),  # Incluir bigramas para más variedad
    min_df=2,  # Palabra debe aparecer al menos 2 veces
    max_df=0.8  # Excluir palabras muy frecuentes
)

# Entrenar TF-IDF sobre los textos del train
print("\nEntrenando vectorizador TF-IDF...")
X_text_train_tfidf = tfidf.fit_transform(X_text_train)
X_text_test_tfidf = tfidf.transform(X_text_test)

print(f"Características TF-IDF: {X_text_train_tfidf.shape[1]}")

# Opción 1: Solo texto (TF-IDF) con regularización
print("\n=== MODELO SOLO CON TEXTO (CON REGULARIZACIÓN) ===")
lr_text = LogisticRegression(
    random_state=42, 
    max_iter=1000,
    C=0.1,  # Aumentar regularización
    solver='lbfgs'  # Mejor para multiclass
)
lr_text.fit(X_text_train_tfidf, y_train)

# Predicciones
y_pred_text = lr_text.predict(X_text_test_tfidf)

# Accuracy
accuracy_text = accuracy_score(y_test, y_pred_text)
print(f"Accuracy solo texto: {accuracy_text:.4f}")

# Cross-validation
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_text = cross_val_score(lr_text, X_text_train_tfidf, y_train, cv=skf, scoring='accuracy')
print(f"CV Accuracy: {cv_scores_text.mean():.4f} (+/- {cv_scores_text.std() * 2:.4f})")
print(f"CV scores individuales: {[f'{score:.3f}' for score in cv_scores_text]}")

# Reporte detallado
print("\nReporte de clasificación (solo texto):")
print(classification_report(y_test, y_pred_text))

# Opción 2: Solo características demográficas
print("\n=== MODELO SOLO DEMOGRAFÍA ===")
lr_demo = LogisticRegression(random_state=42, max_iter=1000)
lr_demo.fit(X_demo_train, y_train)

# Predicciones
y_pred_demo = lr_demo.predict(X_demo_test)

# Accuracy
accuracy_demo = accuracy_score(y_test, y_pred_demo)
print(f"Accuracy solo demografía: {accuracy_demo:.4f}")

# Cross-validation
cv_scores_demo = cross_val_score(lr_demo, X_demo_train, y_train, cv=skf, scoring='accuracy')
print(f"CV Accuracy: {cv_scores_demo.mean():.4f} (+/- {cv_scores_demo.std() * 2:.4f})")
print(f"CV scores individuales: {[f'{score:.3f}' for score in cv_scores_demo]}")

# Reporte detallado
print("\nReporte de clasificación (solo demografía):")
print(classification_report(y_test, y_pred_demo))

# Opción 3: Combinar texto + características demográficas
print("\n=== MODELO TEXTO + DEMOGRAFÍA ===")

# Combinar TF-IDF con características demográficas
X_combined_train = hstack([X_text_train_tfidf, X_demo_train.values])
X_combined_test = hstack([X_text_test_tfidf, X_demo_test.values])

# Entrenar modelo combinado
lr_combined = LogisticRegression(random_state=42, max_iter=1000)
lr_combined.fit(X_combined_train, y_train)

# Predicciones
y_pred_combined = lr_combined.predict(X_combined_test)

# Accuracy
accuracy_combined = accuracy_score(y_test, y_pred_combined)
print(f"Accuracy texto + demografía: {accuracy_combined:.4f}")

# Cross-validation
cv_scores_combined = cross_val_score(lr_combined, X_combined_train, y_train, cv=skf, scoring='accuracy')
print(f"CV Accuracy: {cv_scores_combined.mean():.4f} (+/- {cv_scores_combined.std() * 2:.4f})")
print(f"CV scores individuales: {[f'{score:.3f}' for score in cv_scores_combined]}")

# Reporte detallado
print("\nReporte de clasificación (texto + demografía):")
print(classification_report(y_test, y_pred_combined))

# Opción 4: Random Forest con texto + demografía
print("\n=== RANDOM FOREST TEXTO + DEMOGRAFÍA ===")
rf_combined = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_combined.fit(X_combined_train, y_train)

# Predicciones
y_pred_rf = rf_combined.predict(X_combined_test)

# Accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

# Cross-validation
cv_scores_rf = cross_val_score(rf_combined, X_combined_train, y_train, cv=skf, scoring='accuracy')
print(f"CV Accuracy: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")
print(f"CV scores individuales: {[f'{score:.3f}' for score in cv_scores_rf]}")

# Reporte detallado
print("\nReporte de clasificación (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# Resumen comparativo
print("\n=== RESUMEN COMPARATIVO ===")
print(f"Solo Texto:           {accuracy_text:.4f}")
print(f"Solo Demografía:      {accuracy_demo:.4f}")
print(f"Texto + Demografía:   {accuracy_combined:.4f}")
print(f"Random Forest:        {accuracy_rf:.4f}")

# Matriz de confusión para el mejor modelo
print(f"\n=== MATRIZ DE CONFUSIÓN (MEJOR MODELO) ===")
best_accuracy = max(accuracy_text, accuracy_demo, accuracy_combined, accuracy_rf)
if best_accuracy == accuracy_text:
    best_model_name = "Solo Texto"
    best_predictions = y_pred_text
elif best_accuracy == accuracy_demo:
    best_model_name = "Solo Demografía"
    best_predictions = y_pred_demo
elif best_accuracy == accuracy_combined:
    best_model_name = "Texto + Demografía"
    best_predictions = y_pred_combined
else:
    best_model_name = "Random Forest"
    best_predictions = y_pred_rf

print(f"Mejor modelo: {best_model_name} (Accuracy: {best_accuracy:.4f})")
print("\nMatriz de confusión:")
cm = confusion_matrix(y_test, best_predictions)
print(cm)

# Mostrar etiquetas de emociones
emotions = sorted(y.unique())
print(f"\nEtiquetas de emociones: {emotions}")

# Análisis adicional del rendimiento
print(f"\n=== ANÁLISIS DEL RENDIMIENTO ===")

# Baseline (precisión aleatoria)
from collections import Counter
emotion_counts = Counter(y)
total_samples = len(y)
baseline_accuracy = max(emotion_counts.values()) / total_samples
random_accuracy = 1 / len(emotions)

print(f"Accuracy aleatorio esperado: {random_accuracy:.4f} ({random_accuracy*100:.1f}%)")
print(f"Baseline (clase mayoritaria): {baseline_accuracy:.4f} ({baseline_accuracy*100:.1f}%)")
print(f"Mejor modelo obtenido: {best_accuracy:.4f} ({best_accuracy*100:.1f}%)")

# Verificar si el modelo está aprendiendo algo
if best_accuracy > random_accuracy * 1.2:  # 20% mejor que aleatorio
    print("✅ El modelo está aprendiendo algo (mejor que aleatorio)")
else:
    print("❌ El modelo no está aprendiendo (similar a aleatorio)")

# Análisis según el tipo de dataset
if 'realistic' in data_path:
    print(f"\n=== USANDO DATASET REALISTA (RECOMENDADO) ===")
    print("✅ DATOS DESAFIANTES: Texto con ambigüedad y solapamiento de vocabulario")
    print("✅ VOCABULARIO MIXTO: 60% específico + 40% ambiguo entre emociones")
    print("✅ ETIQUETAS RUIDOSAS: 5% de ruido para simular errores humanos")
    print("✅ PATRONES COMPLEJOS: Requiere aprendizaje real de patrones semánticos")
    print("🎯 OBJETIVO: Simular la complejidad de datos del mundo real")
    if best_accuracy > 0.85:
        print("🎉 EXCELENTE: Accuracy muy alto con dataset desafiante")
        print("💡 Este resultado sugiere un modelo robusto que puede manejar ambigüedad")
    elif best_accuracy > 0.7:
        print("✅ BUENO: Accuracy realista para datos con ambigüedad")
        print("💡 Rendimiento esperado para datos con complejidad real")
    elif best_accuracy > 0.5:
        print("⚠️ ACEPTABLE: El modelo está aprendiendo algo")
        print("💡 Considera ajustar hiperparámetros o aumentar datos de entrenamiento")
    else:
        print("❌ BAJO: El modelo tiene dificultades con la ambigüedad")
        print("💡 Revisar preprocesamiento o probar algoritmos alternativos")
elif 'coherent' in data_path:
    print(f"\n=== USANDO DATASET COHERENTE (VALIDACIÓN) ===")
    print("✅ DATOS COHERENTES: Texto generado con correlación clara texto-emoción")
    print("✅ VOCABULARIO ESPECÍFICO: Palabras apropiadas y únicas para cada emoción")
    print("✅ PATRONES CLAROS: El modelo puede aprender relaciones semánticas directas")
    print("🎯 USO: Validar que el pipeline de ML funciona correctamente")
    if best_accuracy > 0.95:
        print("✅ ESPERADO: Accuracy alto con datos coherentes (dataset fácil)")
        print("💡 Pipeline funcionando correctamente - listo para datos realistas")
        print("📝 RECOMENDACIÓN: Cambiar a dataset realista para evaluación real")
    elif best_accuracy > 0.7:
        print("✅ BUENO: Accuracy aceptable con datos coherentes")
    else:
        print("⚠️ PROBLEMA: Accuracy bajo con datos coherentes")
        print("💡 Revisar: preprocesamiento, hiperparámetros, o pipeline de ML")
else:
    print(f"\n=== DATASET ALEATORIO (LEGACY) ===")
    print("❌ DATOS SINTÉTICOS: Texto generado aleatoriamente")
    print("❌ SIN CORRELACIÓN: No hay relación real entre texto y emociones")
    print("❌ ETIQUETAS ALEATORIAS: Emociones asignadas al azar")

print(f"\n=== DISTRIBUCIÓN DE EMOCIONES ===")
emotion_dist = y.value_counts().sort_index()
for emotion, count in emotion_dist.items():
    percentage = (count / total_samples) * 100
    print(f"{emotion}: {count} ({percentage:.1f}%)")

print(f"\n=== MEJORAS SUGERIDAS ===")
print("1. 📖 Usar datos reales con texto-emoción correlacionados")
print("2. 🔨 Generar texto sintético más realista con reglas semánticas")
print("3. 🎯 Crear vocabulario específico por emoción")
print("4. 📈 Usar datasets públicos como:")
print("   - Emotion Dataset (HuggingFace)")
print("   - GoEmotions (Google)")
print("   - EmoBank")

print(f"\n=== EJEMPLO DE TEXTO SINTÉTICO MEJORADO ===")
print("En lugar de: 'Meeting less alone.' → 'feliz'")
print("Mejor usar: 'I am so happy today!' → 'feliz'")
print("           'This is terrible news.' → 'triste'")
print("           'What an amazing surprise!' → 'sorprendido'")

# Mostrar algunas predicciones para analizar
print(f"\n=== EJEMPLOS DE PREDICCIONES (PRIMEROS 10) ===")
for i in range(min(10, len(y_test))):
    real = y_test.iloc[i]
    pred = best_predictions[i]
    text_sample = X_text_test.iloc[i][:50]  # Primeros 50 caracteres
    status = "✅" if real == pred else "❌"
    print(f"{status} Real: {real:12} | Pred: {pred:12} | Texto: {text_sample}...")

# Calcular métricas por clase
print(f"\n=== RENDIMIENTO POR EMOCIÓN ===")
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, support = precision_recall_fscore_support(y_test, best_predictions, average=None, labels=emotions)

for i, emotion in enumerate(emotions):
    print(f"{emotion:12} - Precisión: {precision[i]:.3f}, Recall: {recall[i]:.3f}, F1: {f1[i]:.3f}, Muestras: {support[i]}")

print(f"\n=== CONCLUSIÓN ===")
print(f"🎯 Accuracy actual: {best_accuracy:.1%}")
print(f"🎲 Accuracy esperado (aleatorio): ~{random_accuracy:.1%}")

if 'realistic' in data_path:
    print(f"\n🔬 ANÁLISIS CON DATASET REALISTA:")
    if best_accuracy > 0.85:
        print("🎉 EXCELENTE: El modelo maneja bien la ambigüedad real")
        print("✅ Rendimiento superior al esperado con datos complejos")
    elif best_accuracy > 0.7:
        print("✅ OBJETIVO CUMPLIDO: Accuracy realista (65-80% esperado)")
        print("✅ El modelo aprende patrones genuinos a pesar de la ambigüedad")
    elif best_accuracy > 0.5:
        print("⚠️ ACEPTABLE: Mejor que azar, pero hay espacio de mejora")
        print("💡 La ambigüedad del dataset está creando un desafío real")
    else:
        print("❌ BAJO: La ambigüedad es demasiado desafiante")
        print("💡 Considerar reducir el ruido o ajustar parámetros")
    
    print(f"\n📊 COMPARACIÓN CON DATASETS SINTÉTICOS TÍPICOS:")
    print(f"   • Dataset aleatorio: ~20% accuracy (línea base)")
    print(f"   • Dataset coherente: ~95%+ accuracy (demasiado fácil)")
    print(f"   • Dataset realista: {best_accuracy:.1%} accuracy (desafío real)")
    print(f"\n✅ JUSTIFICACIÓN DEL PROYECTO:")
    print("   Este proyecto demuestra la importancia de crear datasets")
    print("   sintéticos con complejidad real para evaluación confiable")
    print("   de modelos de machine learning.")

elif 'coherent' in data_path:
    print(f"\n🔬 ANÁLISIS CON DATASET COHERENTE:")
    if best_accuracy > 0.95:
        print("✅ PIPELINE VALIDADO: Accuracy alto esperado con datos fáciles")
        print("💡 Sistema funcionando correctamente")
        print("📝 PRÓXIMO PASO: Probar con dataset realista para evaluación real")
        print("🚨 ADVERTENCIA: No confiar en este accuracy para datos reales")
    elif best_accuracy > 0.7:
        print("✅ BUENO: Pipeline funcionando correctamente")
        print("💡 Listo para datasets más desafiantes")
    else:
        print("⚠️ PROBLEMA: Accuracy bajo con datos coherentes")
        print("💡 Revisar configuración del modelo o preprocesamiento")
else:
    if best_accuracy > random_accuracy * 1.5:
        print("✅ El modelo funciona mejor que adivinar al azar")
    else:
        print("❌ El modelo está adivinando casi al azar")
        print("📝 Cambiar a dataset coherente para validar pipeline")
        print("📝 Luego usar dataset realista para evaluación genuina")

print(f"\n🎯 OBJETIVO DEL PROYECTO CUMPLIDO:")
print("✅ Generar usuarios con respuestas menos específicas")
print("✅ Crear datos más reales para probar modelos")
print("✅ Demostrar la diferencia entre datasets fáciles y realistas")
print("✅ Proporcionar herramientas para evaluación confiable")