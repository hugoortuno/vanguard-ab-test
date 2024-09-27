import os
from functions import cargar_datos, inspeccionar_datos

# Definir la carpeta donde están almacenados los archivos
carpeta = 'data/'

# Nombres de los archivos
archivos = {
    "df_final_demo": "df_final_demo.txt",
    "df_final_experiment_clients": "df_final_experiment_clients.txt",
    "df_final_web_data_pt_1": "df_final_web_data_pt_1.txt",
    "df_final_web_data_pt_2": "df_final_web_data_pt_2.txt"
}

# Cargar los archivos en dataframes
df_demo = cargar_datos(os.path.join(carpeta, archivos['df_final_demo']))
df_experiment_clients = cargar_datos(os.path.join(carpeta, archivos['df_final_experiment_clients']))
df_web_pt_1 = cargar_datos(os.path.join(carpeta, archivos['df_final_web_data_pt_1']))
df_web_pt_2 = cargar_datos(os.path.join(carpeta, archivos['df_final_web_data_pt_2']))

# Inspeccionar los archivos cargados
inspeccionar_datos(df_demo, "df_final_demo")
inspeccionar_datos(df_experiment_clients, "df_final_experiment_clients")
inspeccionar_datos(df_web_pt_1, "df_final_web_data_pt_1")
inspeccionar_datos(df_web_pt_2, "df_final_web_data_pt_2")

