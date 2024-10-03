# Importar el módulo de funciones (asegúrate de que el archivo de funciones se llama functions.py)
from functions import procesar_dataframe  # Importa la función de procesamiento del segundo bloque
from functions import crear_carpetas
from functions import cargar_y_preparar_datos
from functions import verificar_valores_atipicos
from functions import tratamiento_valores_nulos
from functions import conversion_de_tipos
from functions import analisis_exploratorio
from functions import modelo_predictivo
from functions import agregar_rango_antiguedad_y_balance
from functions import agregar_metricas
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def main():
    crear_carpetas()
    df = cargar_y_preparar_datos()
    verificar_valores_atipicos(df)
    df = tratamiento_valores_nulos(df)
    df = conversion_de_tipos(df)
    analisis_exploratorio(df)
    modelo_predictivo(df)
    df = agregar_rango_antiguedad_y_balance(df)
    agregar_metricas(df)

    # Cargar los dos archivos y procesar el DataFrame combinado
    df_pt1 = pd.read_csv('data/df_final_web_data_pt_1.txt', delimiter=',')
    df_pt2 = pd.read_csv('data/df_final_web_data_pt_2.txt', delimiter=',')
    df_combined = pd.concat([df_pt1, df_pt2], ignore_index=True)
    
    procesar_dataframe(df_combined, 'cleaned_df_final_web_data')

    # Cargar y procesar el archivo de clientes de experimento
    file_path = 'data/df_final_experiment_clients.txt'
    df_experiment = pd.read_csv(file_path, delimiter=',')
    df_experiment.rename(columns={'client_id': 'Identificador del cliente', 'Variation': 'Variación'}, inplace=True)
    
    variacion_counts = df_experiment['Variación'].value_counts()
    plt.bar(variacion_counts.index, variacion_counts.values, color=['blue', 'orange'])
    plt.title('Distribución de la Variación')
    plt.xlabel('Variación')
    plt.ylabel('Número de Clientes')
    plt.xticks(rotation=0)
    plt.show()

    output_file_path_experiment = 'cleaned_data/cleaned_df_final_experiment_clients.xlsx'
    df_experiment.to_excel(output_file_path_experiment, index=False)
    print(f"\nArchivo guardado como: {output_file_path_experiment}")

if __name__ == '__main__':
    main()
