import pandas as pd

# Función para cargar los datos desde archivos .txt con delimitador ',' desde la carpeta 'data'
def cargar_datos(ruta_archivo):
    """Cargar datos de un archivo .txt con delimitador ','."""
    return pd.read_csv(ruta_archivo, delimiter=',')

# Función para inspeccionar los primeros valores de cada archivo
def inspeccionar_datos(df, nombre):
    """Mostrar las primeras filas del dataframe y la información general."""
    print(f"Mostrando los primeros valores de {nombre}:\n")
    print(df.head())
    print("\nInformación general:\n")
    print(df.info())
    print("\nValores faltantes:\n")
    print(df.isnull().sum())
    print("="*50)

# Función para fusionar datos de clientes experimentales
def fusionar_datos_experimentales(df_parte1, df_parte2):
    """Fusionar dos partes de los datos web en un solo dataframe."""
    return pd.concat([df_parte1, df_parte2], axis=0)

import pandas as pd
import numpy as np

def limpiar_datos(df_final_demo, df_final_experiment_clients, df_final_web_data_pt_1, df_final_web_data_pt_2):
    # Limpiar df_final_demo
    # Eliminar o imputar valores nulos según sea necesario
    df_final_demo['gendr'] = df_final_demo['gendr'].replace('U', np.nan)
    
    # Imputar valores faltantes en num_accts y otros valores numéricos con la mediana
    for col in ['clnt_tenure_yr', 'clnt_tenure_mnth', 'clnt_age', 'num_accts', 'bal', 'calls_6_mnth', 'logons_6_mnth']:
        df_final_demo[col].fillna(df_final_demo[col].median(), inplace=True)
    
    # Imputar género con el valor más frecuente
    df_final_demo['gendr'].fillna(df_final_demo['gendr'].mode()[0], inplace=True)
    
    # Limpiar df_final_experiment_clients
    # Eliminar filas con valores nulos en la columna 'Variation' o imputar con un valor como 'Unknown'
    df_final_experiment_clients['Variation'].fillna('Unknown', inplace=True)

    # Convertir columnas date_time a formato datetime en los dos dataframes web
    df_final_web_data_pt_1['date_time'] = pd.to_datetime(df_final_web_data_pt_1['date_time'])
    df_final_web_data_pt_2['date_time'] = pd.to_datetime(df_final_web_data_pt_2['date_time'])
    
    # Concatenar los dos dataframes web
    df_final_web_data = pd.concat([df_final_web_data_pt_1, df_final_web_data_pt_2], ignore_index=True)
    
    return df_final_demo, df_final_experiment_clients, df_final_web_data

import pandas as pd

# Unir df_final_demo con df_final_experiment_clients
df_merged_demo_experiment = pd.merge(df_final_demo, df_final_experiment_clients, on='client_id', how='left')

# Verificar la unión
print(df_merged_demo_experiment.head())
