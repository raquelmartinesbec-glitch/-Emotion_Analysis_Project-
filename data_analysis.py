import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar el nuevo dataset limpio de 1000 usuarios
data_clean_train = pd.read_csv('data/data_clean_emotions_1000.csv')

# Opción 1: Usar solo características demográficas y temporales (como antes)
X_basic = data_clean_train[['age', 'year', 'month', 'day', 'day_of_week', 'hour',
                           'gender_masculino', 'gender_no binario', 
                           'region_Este', 'region_Norte', 'region_Oeste', 'region_Sur']]

# Opción 2: Usar todas las características numéricas (incluyendo TF-IDF del texto)
# Excluir solo las columnas no numéricas o de identificación
exclude_cols = ['user_id', 'timestamp', 'text', 'text_clean', 'emotion']
X_full = data_clean_train.drop(columns=exclude_cols)

# Selecciona cual usar (cambia X_basic por X_full si quieres incluir el texto)
X = X_basic  # O usa X = X_full para todas las características

y = data_clean_train["emotion"]  # columna objetivo

print("Dataset emociones cargado. Número de registros: {}".format(X.shape[0]))
print("Número de características: {}".format(X.shape[1]))

# Dividimos en 20% para test y 80% para train
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=20,
    stratify=y  # Mantener proporción de clases
)

print("Hay {} registros en el set de train y {} registros en el set de test".format(
    X_train.shape[0], X_test.shape[0]
))

print("\nDistribución de emociones en train:")
print(y_train.value_counts())