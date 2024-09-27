import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def clean_demo(df):
    # Verificar valores nulos
    print(df.isnull().sum())
    
    # Normalizar valores de género
    df['gendr'] = df['gendr'].str.strip().str.lower()
    
    # Convertir 'bal' a tipo numérico
    df['bal'] = pd.to_numeric(df['bal'], errors='coerce')
    
    # Verificar datos duplicados
    df.drop_duplicates(inplace=True)
    
    return df

def clean_experiment(df, df_demo):
    # Verificar valores nulos
    print(df.isnull().sum())
    
    # Asegurarse de que todos los 'client_id' están en df_demo
    if not df['client_id'].isin(df_demo['client_id']).all():
        raise ValueError("Some client_ids in df are not in df_demo")
    
    return df

def clean_web_data(df1, df2):
    # Unir los dos DataFrames
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Convertir 'date_time' a datetime
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # Crear nuevas columnas basadas en 'date_time'
    df['Año'] = df['date_time'].dt.year
    df['Mes'] = df['date_time'].dt.month_name().str.capitalize()
    df['Día'] = df['date_time'].dt.day_name().str.capitalize()
    df['Fecha'] = df['date_time'].dt.day
    
    # Eliminar duplicados
    df.drop_duplicates(inplace=True)
    
    return df

def demographic_analysis(df):
    print("Distribución por género:")
    print(df['gendr'].value_counts())
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df['clnt_age'].dropna(), bins=20, kde=True)
    plt.title('Distribución de Edad de los Clientes')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.show()

def behavior_analysis(df):
    # Por ejemplo, calcular el promedio de tiempo en cada paso
    df['step_duration'] = df.groupby('client_id')['date_time'].diff().fillna(pd.Timedelta(seconds=0))
    avg_step_duration = df.groupby('process_step')['step_duration'].mean()
    print(avg_step_duration)

def calculate_kpis(df):
    # Calcular tasa de finalización
    completion_rate = df[df['process_step'] == 'confirm'].shape[0] / df['client_id'].nunique()
    
    # Calcular tiempo promedio por paso
    avg_time_per_step = df.groupby('process_step')['step_duration'].mean()
    
    return completion_rate, avg_time_per_step

def test_hypothesis(df_control, df_test):
    # Calcular el número de confirmaciones y total de usuarios
    control_total = df_control['client_id'].nunique()
    test_total = df_test['client_id'].nunique()
    control_confirmations = df_control[df_control['process_step'] == 'confirm'].shape[0]
    test_confirmations = df_test[df_test['process_step'] == 'confirm'].shape[0]
    
    # Construir la tabla de contingencia
    contingency_table = [
        [control_confirmations, control_total - control_confirmations],
        [test_confirmations, test_total - test_confirmations]
    ]
    
    # Realizar la prueba chi-cuadrado
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
    
    # Calcular tasas de finalización
    control_completion = control_confirmations / control_total
    test_completion = test_confirmations / test_total
    
    return control_completion, test_completion, p_value
