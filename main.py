import pandas as pd
from functions import (
    clean_demo, 
    clean_experiment, 
    clean_web_data, 
    demographic_analysis, 
    behavior_analysis, 
    calculate_kpis, 
    test_hypothesis
)

def main():
    # Cargar los datos
    df_demo = pd.read_csv('D:/Documents/GitHub/vanguard-ab-test/data/df_final_demo.txt', delimiter=',')
    df_experiment = pd.read_csv('D:/Documents/GitHub/vanguard-ab-test/data/df_final_experiment_clients.txt', delimiter=',')
    df1 = pd.read_csv('D:/Documents/GitHub/vanguard-ab-test/data/df_final_web_data_pt_1.txt', delimiter=',')
    df2 = pd.read_csv('D:/Documents/GitHub/vanguard-ab-test/data/df_final_web_data_pt_2.txt', delimiter=',')

    
    # Limpieza de datos
    df_demo_clean = clean_demo(df_demo)
    df_experiment_clean = clean_experiment(df_experiment, df_demo_clean)
    df_web_clean = clean_web_data(df1, df2)
    
    # Análisis Demográfico
    demographic_analysis(df_demo_clean)
    
    # Análisis de Comportamiento
    behavior_analysis(df_experiment_clean)
    
    # Cálculo de KPIs
    completion_rate, avg_time_per_step = calculate_kpis(df_experiment_clean)
    print(f"Tasa de Finalización: {completion_rate:.2f}")
    print("Tiempo Promedio por Paso:")
    print(avg_time_per_step)
    
    # Prueba de Hipótesis
    df_control = df_experiment_clean[df_experiment_clean['group'] == 'control']
    df_test = df_experiment_clean[df_experiment_clean['group'] == 'test']
    control_completion, test_completion, p_value = test_hypothesis(df_control, df_test)
    
    print(f"Tasa de Finalización (Control): {control_completion:.2f}")
    print(f"Tasa de Finalización (Test): {test_completion:.2f}")
    print(f"Valor p de la prueba chi-cuadrado: {p_value:.4f}")
    
    # Conclusiones
    if p_value < 0.05:
        print("La diferencia entre los grupos es estadísticamente significativa.")
    else:
        print("No hay evidencia suficiente para afirmar que la diferencia entre los grupos es significativa.")

if __name__ == "__main__":
    main()
