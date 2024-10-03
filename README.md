# Proyecto: Vanguard A/B Test

## Introducción

Bienvenido al proyecto de pruebas A/B de Vanguard. En este proyecto, aplicaremos habilidades de análisis de datos, limpieza, análisis exploratorio, métricas de rendimiento, pruebas de hipótesis y evaluaciones experimentales.

Vanguard, una empresa de gestión de inversiones, ha realizado un experimento para mejorar la experiencia del usuario mediante una nueva interfaz de usuario (UI). El objetivo principal es determinar si esta nueva UI mejora las tasas de finalización de procesos online.

## Contexto

El experimento se llevó a cabo del 15 de marzo de 2017 al 20 de junio de 2017 y se dividió en dos grupos:

- **Grupo Control**: Usuarios que interactuaron con la interfaz tradicional de Vanguard.
- **Grupo Test**: Usuarios que probaron la nueva UI mejorada.

Ambos grupos completaron el mismo proceso, que incluía una página de inicio, tres pasos intermedios y una página de confirmación.

### Pregunta clave:

**¿La nueva interfaz mejoró las tasas de finalización?**

## Datasets

Los datos proporcionados se dividen en tres conjuntos principales:

1. **Perfiles de los clientes (df_final_demo)**: [Datos demográficos de los clientes como edad, género y detalles de cuenta.](https://github.com/data-bootcamp-v4/lessons/blob/main/5_6_eda_inf_stats_tableau/project/files_for_project/df_final_demo.txt)
  
2. **Huella Digital (df_final_web_data)**: Registros de interacciones en línea, divididos en dos partes que deben combinarse antes del análisis.

   - [Parte 1:](https://github.com/data-bootcamp-v4/lessons/blob/main/5_6_eda_inf_stats_tableau/project/files_for_project/df_final_web_data_pt_1.txt)
   - [Parte 2:](https://github.com/data-bootcamp-v4/lessons/blob/main/5_6_eda_inf_stats_tableau/project/files_for_project/df_final_web_data_pt_2.txt)

3. **Clientes del experimento (df_final_experiment_clients)**: [Lista de clientes que participaron en el experimento.](https://github.com/data-bootcamp-v4/lessons/blob/main/5_6_eda_inf_stats_tableau/project/files_for_project/df_final_experiment_clients.txt)

## Configuración

Este proyecto se gestiona mediante un [tablero Kanban en Trello:](https://trello.com/b/EZUEnHaq/vanguard-a-b-test).

## Estructura del proyecto

El proyecto se ha dividido en las siguientes etapas:

1. **Análisis Exploratorio y Limpieza de Datos (EDA)**

   - Exploración y limpieza de los datos.
   - Análisis del comportamiento de los clientes.
   
2. **Métricas de rendimiento**

   - Definición de KPI para evaluar la nueva interfaz: tasa de finalización, tiempo por paso, tasa de errores.
   
3. **Pruebas de hipótesis**

   - Evaluación de la tasa de finalización del grupo de prueba comparada con el grupo de control.

4. **Evaluación del experimento**

   - Revisión del diseño del experimento y la duración.

5. **Visualizaciones en Power BI**

   - Creación de visualizaciones interactivas. [Enlace a Power BI](https://app.powerbi.com/groups/me/reports/b64337b3-e815-4c3f-8dc8-e2c9d01b12bd/9d33cc8bf955c5a098ec?experience=power-bi).

## Preguntas clave

- ¿Quiénes son los principales usuarios del proceso en línea?
- ¿Qué patrones de comportamiento de clientes se pueden identificar?

## Hipótesis

1. **Tasa de finalización**: El grupo de prueba (TEST) mejorará la tasa de finalización respecto al grupo de control (CONTROL).
2. **Rentabilidad**: La nueva interfaz será rentable si mejora un umbral predeterminado.

## Conclusión

Al final del análisis, determinaremos si el rediseño de la interfaz es efectivo.

## Repositorio

El código del proyecto está disponible en: [GitHub](https://github.com/hugoortuno/vanguard-ab-test).

## Presentación

Puedes revisar la presentación en [Canva](https://www.canva.com/design/DAGShL6sa34/pUpFJJv2Mg8v0Fr_yoBkog/edit?utm_content=DAGShL6sa34&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).
