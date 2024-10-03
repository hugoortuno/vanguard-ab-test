# Bloque 1

# Importar las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os


def crear_carpetas():
    # Crear las carpetas cleaned_data y views si no existen
    os.makedirs('cleaned_data', exist_ok=True)
    os.makedirs('views', exist_ok=True)


def cargar_y_preparar_datos():
    # Cargar los datos desde el archivo de texto
    df = pd.read_csv('data/df_final_demo.txt', delimiter=',')
    # Renombrar las columnas a español
    df.columns = [
        'Identificador del cliente', 
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


def agregar_rango_antiguedad_y_balance(df):
    bins = [0, 18, 25, 35, 45, 55, 65, float('inf')]  # Definir los rangos de antigüedad
    labels = ['1.Menores de edad', '2.18-25 años', '3.25-35 años', '4.35-45 años', '5.45-55 años', '6.55-65 años', '7.65+ años']
    df['Rango de Edad'] = pd.cut(df['Edad'], bins=bins, labels=labels, right=False)

    bins = [0, 3, 5, 10, 15, 20, 25, 30, float('inf')]  # Definir los rangos de antigüedad
    labels = ['1.Menos de 3 años', '2.3-5 años', '3.5-10 años', '4.10-15 años', '5.15-20 años', '6.20-25 años', '7.25-30 años', '8.30+ años']
    df['Rango de antigüedad'] = pd.cut(df['Antigüedad en años'], bins=bins, labels=labels, right=False)

    bins = [0, 10000, 20000, 50000, 100000, 250000, 500000, 1000000, 5000000, float('inf')]  # Definir los rangos por Saldo
    labels = ['1.<10 mil', '2.10-20 mil', '3.20-50 mil', '4.50-100 mil', '5.100-250 mil', '6.250-500 mil', '7.500 mil-1 millón', '8.1-5 milones', '9.5+ millones']
    df['Rango de saldo'] = pd.cut(df['Saldo'], bins=bins, labels=labels, right=False)
  
    return df


def agregar_metricas(df):
    df['Media de saldo'] = df['Saldo'].mean()
    df['Desviación Estándar del saldo'] = df['Saldo'].std()
    df['Saldo mínimo'] = df['Saldo'].min()
    df['Saldo máximo'] = df['Saldo'].max()
    df['Total de clientes'] = len(df)

    promedio_antiguedad = df.groupby('Género', observed=True)['Antigüedad en años'].mean().rename('Promedio de antigüedad por género')
    df = df.join(promedio_antiguedad, on='Género')

    saldo_columns = ['Media de saldo', 'Desviación Estándar del saldo', 'Saldo mínimo', 'Saldo máximo']
    for col in saldo_columns:
        df[col] = df[col].apply(lambda x: f'€{x:,.2f}')

    output_file_path = 'cleaned_data/cleaned_df_final_demo.xlsx'
    df.to_excel(output_file_path, index=False)
    print(f'Datos limpios y métricas agregadas guardados en: {output_file_path}')


def procesar_dataframe(df, nombre_archivo):
    # Renombrar las columnas
    df.columns = ['Identificador del cliente', 'Identificador del visitante', 'Identificador de la visita', 'Pasos del proceso', 'Fecha y hora']
    
    # Convertir la columna de fecha y hora
    df['Fecha y hora'] = pd.to_datetime(df['Fecha y hora'])
    df['Año'] = df['Fecha y hora'].dt.year
    df['Mes'] = df['Fecha y hora'].dt.month_name(locale='es_ES').str.capitalize()
    df['Día'] = df['Fecha y hora'].dt.day_name(locale='es_ES').str.capitalize()
    df['Fecha'] = df['Fecha y hora'].dt.day

    # Renombrar los valores en la columna 'Pasos del proceso'
    renombrar_pasos = {
        'step_3': '3.Paso 3',
        'step_2': '2.Paso 2',
        'step_1': '1.Paso 1',
        'start': '0.Inicio',
        'confirm': '9.Confirmación'
    }
    df['Pasos del proceso'] = df['Pasos del proceso'].replace(renombrar_pasos)

    # Mostrar valores únicos
    print(f"Valores únicos en 'Pasos del proceso': {df['Pasos del proceso'].unique()}")

    # Eliminar valores NaN
    df_cleaned = df.dropna()

    # Ordenar el DataFrame por 'Identificador de la visita' y 'Fecha y hora' por si acaso no están en orden
    df_cleaned = df_cleaned.sort_values(by=['Identificador de la visita', 'Fecha y hora'])

    # Desplazar la columna 'Fecha y hora' hacia arriba para obtener la fecha del siguiente paso
    df_cleaned['Fecha y hora siguiente'] = df_cleaned.groupby('Identificador de la visita')['Fecha y hora'].shift(-1)

    # Desplazar la columna 'Pasos del proceso' hacia arriba para obtener el nombre del siguiente paso
    df_cleaned['Paso siguiente'] = df_cleaned.groupby('Identificador de la visita')['Pasos del proceso'].shift(-1)
    
    # Calcular la duración de cada paso
    df_cleaned['Duración del paso'] = df_cleaned['Fecha y hora siguiente'] - df_cleaned['Fecha y hora']

    # Calcular el total de visitas por cliente
    df_cleaned['Total de visitas por cliente'] = df_cleaned.groupby('Identificador del cliente')['Identificador de la visita'].transform('count')
    
    # Calcular la tasa de conversión
    total_visitas = df_cleaned['Identificador de la visita'].count()
    total_confirmaciones = df_cleaned[df_cleaned['Pasos del proceso'] == 'Confirmación']['Identificador de la visita'].count()
    tasa_conversion = (total_confirmaciones / total_visitas) * 100
    df_cleaned['Tasa de conversión'] = tasa_conversion

    # Calcular diferencia de tiempo y tiempo total por cliente
    df_cleaned['Diferencia de tiempo'] = df_cleaned.groupby('Identificador del cliente')['Fecha y hora'].diff().dt.total_seconds()
    df_cleaned['Tiempo total por cliente'] = df_cleaned.groupby('Identificador del cliente')['Diferencia de tiempo'].transform('sum')
    df_cleaned['Tiempo en el paso'] = df_cleaned.groupby(['Identificador del cliente', 'Pasos del proceso'])['Fecha y hora'].diff().dt.total_seconds()

    # Calcular tiempo promedio por paso
    tiempo_promedio_por_paso = df_cleaned.groupby('Pasos del proceso')['Diferencia de tiempo'].mean().reset_index()
    tiempo_promedio_por_paso.columns = ['Pasos del proceso', 'Tiempo promedio por paso en segundos']
    df_cleaned = df_cleaned.merge(tiempo_promedio_por_paso, on='Pasos del proceso', how='left')

    # Calcular el tiempo total en el proceso
    df_cleaned['Tiempo total en proceso'] = df_cleaned.groupby('Identificador del cliente')['Diferencia de tiempo'].cumsum()

    # Calcular el total de clientes únicos
    total_clientes_unicos = df_cleaned['Identificador del cliente'].nunique()
    df_cleaned['Total de clientes únicos'] = total_clientes_unicos

    # Calcular la tasa de retención
    clientes_regresados = df_cleaned[df_cleaned.duplicated(subset='Identificador del cliente', keep=False)]['Identificador del cliente'].nunique()
    tasa_retencion = (clientes_regresados / total_clientes_unicos) * 100
    df_cleaned['Tasa de retención'] = tasa_retencion

    # Calcular el promedio de visitas por cliente
    promedio_visitas_por_cliente = total_visitas / total_clientes_unicos
    df_cleaned['Promedio de visitas por cliente'] = promedio_visitas_por_cliente

    # Guardar el archivo resultante
    output_folder = 'cleaned_data'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = f'{output_folder}/{nombre_archivo}.xlsx'

    if os.path.exists(output_path):
        os.remove(output_path)

    df_cleaned.to_excel(output_path, index=False)
    print(f"El archivo Excel '{nombre_archivo}.xlsx' se ha generado correctamente con las nuevas métricas y KPIs.")

