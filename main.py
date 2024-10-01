# Importar el módulo de funciones (asegúrate de que el archivo de funciones se llama functions.py)
from functions import procesar_dataframe  # Importa la función de procesamiento del segundo bloque
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def crear_carpetas():
    # Crear las carpetas cleaned_data y views si no existen
    os.makedirs('cleaned_data', exist_ok=True)
    os.makedirs('views', exist_ok=True)

def cargar_y_preparar_datos():
    # Cargar los datos desde el archivo de texto
    df = pd.read_csv('data/df_final_demo.txt', delimiter=',')
    # Renombrar las columnas a español
    df.columns = [
        'ID del cliente', 
        'Antigüedad en años', 
        'Antigüedad en meses', 
        'Edad', 
        'Género', 
        'Número de cuentas', 
        'Saldo', 
        'Llamadas en 6 meses', 
        'Logins en 6 meses'
    ]
    return df

def verificar_valores_atipicos(df):
    # Gráfico de caja para el Saldo
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df['Saldo'])
    plt.title('Gráfico de caja para saldo')
    plt.xlabel('Saldo')
    plt.savefig('views/grafico_caja_saldo.png')  # Guardar el gráfico
    plt.close()

    # Gráfico de caja para Antigüedad en años
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df['Antigüedad en años'])
    plt.title('Gráfico de caja para antigüedad en años')
    plt.xlabel('Antigüedad en años')
    plt.savefig('views/grafico_caja_antiguedad.png')  # Guardar el gráfico
    plt.close()

    # Gráfico de caja para Edad
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df['Edad'])
    plt.title('Gráfico de caja para edad')
    plt.xlabel('Edad')
    plt.savefig('views/grafico_caja_edad.png')  # Guardar el gráfico
    plt.close()

def tratamiento_valores_nulos(df):
    df.dropna(inplace=True)
    return df

def conversion_de_tipos(df):
    df['Edad'] = df['Edad'].astype(int)
    df['Género'] = df['Género'].replace({
        'F': 'Mujer', 
        'M': 'Hombre', 
        'Desconocido': 'Indefinido',
        'U': 'Indefinido', 
        'x': 'Indefinido'
    })
    df = df[df['Género'].isin(['Mujer', 'Hombre', 'Indefinido'])]
    df['Género'] = df['Género'].astype('category')
    return df

def analisis_exploratorio(df):
    numeric_df = df.select_dtypes(include='number')
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Matriz de correlación')
    plt.savefig('views/matriz_correlacion.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Antigüedad en años', y='Saldo', data=df)
    plt.title('Antigüedad en años vs saldo')
    plt.xlabel('Antigüedad en años')
    plt.ylabel('Saldo')
    plt.savefig('views/antiguedad_vs_saldo.png')
    plt.close()

def modelo_predictivo(df):
    X = df[['Antigüedad en años', 'Edad', 'Número de cuentas']]
    y = df['Saldo']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mse}')
    print(f'R^2: {r2}')

def agregar_metricas(df):
    df['Media de saldo'] = df['Saldo'].mean()
    df['Desviación Estándar del saldo'] = df['Saldo'].std()
    df['Saldo mínimo'] = df['Saldo'].min()
    df['Saldo máximo'] = df['Saldo'].max()
    df['Total de clientes'] = len(df)

    promedio_antiguedad = df.groupby('Género', observed=True)['Antigüedad en años'].mean().rename('Promedio de antigüedad por género')
    df = df.join(promedio_antiguedad, on='Género')

    saldo_columns = ['Saldo', 'Media de saldo', 'Desviación Estándar del saldo', 'Saldo mínimo', 'Saldo máximo']
    for col in saldo_columns:
        df[col] = df[col].apply(lambda x: f'€{x:,.2f}')

    output_file_path = 'cleaned_data/cleaned_df_final_demo.xlsx'
    df.to_excel(output_file_path, index=False)
    print(f'Datos limpios y métricas agregadas guardados en: {output_file_path}')

def main():
    crear_carpetas()
    df = cargar_y_preparar_datos()
    verificar_valores_atipicos(df)
    df = tratamiento_valores_nulos(df)
    df = conversion_de_tipos(df)
    analisis_exploratorio(df)
    modelo_predictivo(df)
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
