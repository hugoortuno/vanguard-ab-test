# main.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def setup_directories():
    """
    Crea las carpetas necesarias para almacenar datos limpios y visualizaciones.
    """
    os.makedirs('cleaned_data', exist_ok=True)
    os.makedirs('views', exist_ok=True)
    print("Directorios 'cleaned_data' y 'views' asegurados.")

def cargar_datos_block1():
    """
    Carga y renombra las columnas del DataFrame del Bloque 1.
    """
    df = pd.read_csv('data/df_final_demo.txt', delimiter=',')
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
    print("Datos del Bloque 1 cargados y columnas renombradas.")
    return df

def verificar_valores_atipicos(df):
    """
    Genera y guarda gráficos de caja para identificar valores atípicos.
    """
    columnas = ['Saldo', 'Antigüedad en años', 'Edad']
    titulos = ['saldo', 'antigüedad en años', 'edad']
    
    for columna, titulo in zip(columnas, titulos):
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=df[columna])
        plt.title(f'Gráfico de caja para {titulo}')
        plt.xlabel(titulo.capitalize())
        plt.savefig(f'views/grafico_caja_{columna.lower().replace(" ", "_")}.png')
        plt.close()
        print(f"Gráfico de caja para {titulo} guardado.")

def tratar_valores_nulos(df):
    """
    Elimina filas con valores nulos del DataFrame.
    """
    antes = df.shape[0]
    df.dropna(inplace=True)
    despues = df.shape[0]
    print(f"Valores nulos eliminados: {antes - despues} filas eliminadas.")
    return df

def convertir_tipos_datos(df):
    """
    Convierte tipos de datos y mapea valores en la columna 'Género'.
    """
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
    print("Tipos de datos convertidos y valores en 'Género' mapeados.")
    return df

def analisis_exploratorio(df):
    """
    Realiza análisis exploratorio incluyendo matriz de correlación y gráfico de dispersión.
    """
    # Matriz de correlación
    numeric_df = df.select_dtypes(include='number')
    correlation_matrix = numeric_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Matriz de correlación')
    plt.savefig('views/matriz_correlacion.png')
    plt.close()
    print("Matriz de correlación guardada.")
    
    # Gráfico de dispersión
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Antigüedad en años', y='Saldo', data=df)
    plt.title('Antigüedad en años vs saldo')
    plt.xlabel('Antigüedad en años')
    plt.ylabel('Saldo')
    plt.savefig('views/antiguedad_vs_saldo.png')
    plt.close()
    print("Gráfico de dispersión 'Antigüedad en años vs saldo' guardado.")

def modelo_predictivo(df):
    """
    Entrena un modelo de regresión lineal y evalúa su rendimiento.
    """
    X = df[['Antigüedad en años', 'Edad', 'Número de cuentas']]
    y = df['Saldo']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Modelo Predictivo Entrenado. MSE: {mse:.2f}, R^2: {r2:.2f}')

def calcular_metricas(df):
    """
    Calcula nuevas métricas y las agrega al DataFrame, luego guarda el resultado en un archivo Excel.
    """
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
    print(f"Datos limpios y métricas agregadas guardados en: {output_file_path}")

def procesar_dataframe(df, nombre_archivo):
    """
    Procesa un DataFrame específico, calculando métricas y KPIs, y lo guarda en un archivo Excel.
    """
    # Renombrar columnas
    df.columns = ['Identificador del cliente', 'Identificador del visitante', 'Identificador de la visita', 'Pasos del proceso', 'Fecha y hora']
    df['Fecha y hora'] = pd.to_datetime(df['Fecha y hora'])
    df['Año'] = df['Fecha y hora'].dt.year
    df['Mes'] = df['Fecha y hora'].dt.month_name(locale='es_ES').str.capitalize()
    df['Día'] = df['Fecha y hora'].dt.day_name(locale='es_ES').str.capitalize()
    df['Fecha'] = df['Fecha y hora'].dt.day

    renombrar_pasos = {
        'step_3': 'Paso 3',
        'step_2': 'Paso 2',
        'step_1': 'Paso 1',
        'start': 'Inicio',
        'confirm': 'Confirmación'
    }
    df['Pasos del proceso'] = df['Pasos del proceso'].replace(renombrar_pasos)

    print(f"Valores únicos en 'Pasos del proceso' para {nombre_archivo}: {df['Pasos del proceso'].unique()}")

    df_cleaned = df.dropna()

    df_cleaned['Total de visitas por cliente'] = df_cleaned.groupby('Identificador del cliente')['Identificador de la visita'].transform('count')
    total_visitas = df_cleaned['Identificador de la visita'].count()
    total_confirmaciones = df_cleaned[df_cleaned['Pasos del proceso'] == 'Confirmación']['Identificador de la visita'].count()
    tasa_conversion = (total_confirmaciones / total_visitas) * 100
    df_cleaned['Tasa de conversión'] = tasa_conversion

    df_cleaned['Diferencia de tiempo'] = df_cleaned.groupby('Identificador del cliente')['Fecha y hora'].diff().dt.total_seconds()
    df_cleaned['Tiempo total por cliente'] = df_cleaned.groupby('Identificador del cliente')['Diferencia de tiempo'].transform('sum')
    df_cleaned['Tiempo en el paso'] = df_cleaned.groupby(['Identificador del cliente', 'Pasos del proceso'])['Fecha y hora'].diff().dt.total_seconds()

    tiempo_promedio_por_paso = df_cleaned.groupby('Pasos del proceso')['Diferencia de tiempo'].mean().reset_index()
    tiempo_promedio_por_paso.columns = ['Pasos del proceso', 'Tiempo promedio por paso en segundos']
    df_cleaned = df_cleaned.merge(tiempo_promedio_por_paso, on='Pasos del proceso', how='left')

    df_cleaned['Tiempo total en proceso'] = df_cleaned.groupby('Identificador del cliente')['Diferencia de tiempo'].cumsum()

    total_clientes_unicos = df_cleaned['Identificador del cliente'].nunique()
    df_cleaned['Total de clientes únicos'] = total_clientes_unicos

    clientes_regresados = df_cleaned[df_cleaned.duplicated(subset='Identificador del cliente', keep=False)]['Identificador del cliente'].nunique()
    tasa_retencion = (clientes_regresados / total_clientes_unicos) * 100
    df_cleaned['Tasa de retención'] = tasa_retencion

    promedio_visitas_por_cliente = total_visitas / total_clientes_unicos
    df_cleaned['Promedio de visitas por cliente'] = promedio_visitas_por_cliente

    output_folder = 'cleaned_data'
    os.makedirs(output_folder, exist_ok=True)

    output_path = f'{output_folder}/{nombre_archivo}.xlsx'

    if os.path.exists(output_path):
        os.remove(output_path)

    df_cleaned.to_excel(output_path, index=False)
    print(f"El archivo Excel '{nombre_archivo}.xlsx' se ha generado correctamente con las nuevas métricas y KPIs.")

def procesar_block2():
    """
    Procesa los DataFrames del Bloque 2.
    """
    df_pt1 = pd.read_csv('data/df_final_web_data_pt_1.txt', delimiter=',')
    df_pt2 = pd.read_csv('data/df_final_web_data_pt_2.txt', delimiter=',')
    
    procesar_dataframe(df_pt1, 'cleaned_df_final_web_data_pt_1')
    procesar_dataframe(df_pt2, 'cleaned_df_final_web_data_pt_2')

def procesar_block3():
    """
    Procesa el DataFrame del Bloque 3, realiza análisis y guarda los resultados.
    """
    df = pd.read_csv('data/df_final_experiment_clients.txt', delimiter=',')
    
    print("Primeras filas del archivo:")
    print(df.head())
    
    print("\nInformación del DataFrame:")
    print(df.info())
    
    columnas_traducidas = {
        'client_id': 'Identificador del cliente',
        'Variation': 'Variación',
    }
    
    df.rename(columns=columnas_traducidas, inplace=True)
    
    print("\nColumnas renombradas:")
    print(df.columns)
    
    variacion_counts = df['Variación'].value_counts()
    print("\nConteo de valores en la columna 'Variación':")
    print(variacion_counts)
    
    plt.figure(figsize=(8, 6))
    plt.bar(variacion_counts.index, variacion_counts.values, color=['blue', 'orange'])
    plt.title('Distribución de la Variación')
    plt.xlabel('Variación')
    plt.ylabel('Número de Clientes')
    plt.xticks(rotation=0)
    plt.savefig('views/distribucion_variacion.png')  # Guardar el gráfico en lugar de mostrarlo
    plt.close()
    print("Gráfico de distribución de la variación guardado.")
    
    output_file_path = 'cleaned_data/cleaned_df_final_experiment_clients.xlsx'
    df.to_excel(output_file_path, index=False)
    
    print(f"\nArchivo guardado como: {output_file_path}")
    
    # Análisis estadístico opcional
    if 'resultado' in df.columns:
        resultados = df.groupby('Variación')['resultado'].describe()
        print("\nEstadísticas descriptivas por grupo:")
        print(resultados)

def main():
    """
    Función principal que coordina la ejecución de todos los bloques.
    """
    print("Inicio del proceso de análisis de datos.\n")
    
    # Configuración inicial
    setup_directories()
    
    # Bloque 1: Limpieza y análisis de datos
    print("\nProcesando Bloque 1...")
    df_block1 = cargar_datos_block1()
    verificar_valores_atipicos(df_block1)
    df_block1 = tratar_valores_nulos(df_block1)
    df_block1 = convertir_tipos_datos(df_block1)
    analisis_exploratorio(df_block1)
    modelo_predictivo(df_block1)
    calcular_metricas(df_block1)
    
    # Bloque 2: Procesamiento de datos web
    print("\nProcesando Bloque 2...")
    procesar_block2()
    
    # Bloque 3: Análisis de experimentos
    print("\nProcesando Bloque 3...")
    procesar_block3()
    
    print("\nProceso de análisis de datos completado exitosamente.")

if __name__ == "__main__":
    main()
