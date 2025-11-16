import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import re
# import matplotlib.pyplot as plt (Eliminado)
# import seaborn as sns (Eliminado)
import altair as alt
import warnings
from io import BytesIO
import shap


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
# from sklearn.metrics import ConfusionMatrixDisplay (Eliminado)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Análisis de Actividad de Hormigas",
    page_icon="🐜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constantes y Definiciones ---

# Directorio de datos (debe existir en el repositorio de GitHub)
DATA_FOLDER = "datos/"


# Columnas largas originales (como están en el Excel)
# Estas son las columnas que usaremos para validar los archivos subidos.
# (Añadir todas las columnas requeridas si faltan)
COLUMNAS_REQUERIDAS_LARGAS = [
    'hora_inicio', 'hora_fin', 'fecha_hora_sensor', 'battery voltage_última',
    'total_hormigas_entrando_al_nido_en_cargadas',
    'total_hormigas_entrando_al_nido_en_sin_carga',
    'total_hormigas_saliendo_del_nido_sn_cargadas',
    'total_hormigas_saliendo_del_nido_sn_sin_carga',
    'velocidad_promedio_en__cargadas', 'velocidad_promedio_en__sin_carga',
    'velocidad_promedio_sn__cargadas', 'velocidad_promedio_sn__sin_carga',
    'rea_mediana_en__cargadas', 'rea_mediana_en__sin_carga',
    'rea_mediana_sn__cargadas', 'rea_mediana_sn__sin_carga',
    'largo_mediana_en__cargadas', 'largo_mediana_en__sin_carga',
    'largo_mediana_sn__cargadas', 'largo_mediana_sn__sin_carga',
    'ancho_mediana_en__cargadas', 'ancho_mediana_en__sin_carga',
    'ancho_mediana_sn__cargadas', 'ancho_mediana_sn__sin_carga',
    'temp_media', 'temp_max', 'temp_mín', 'dew point_media', 'dew point_mín',
    'solar radiation dgt_media', 'relative humidity_media',
    'relative humidity_max', 'relative humidity_mín', 'precip_total'
]

# Diccionario para renombrar columnas.
# Clave = Nombre largo/original, Valor = Nombre corto/nuevo
RENAMING_DICT = {
    'hora_inicio': 'hora_inicio',
    'hora_fin': 'hora_fin',
    'fecha_hora_sensor': 'fecha_hora_sensor',
    'battery voltage_última': 'bateria_voltaje',
    'total_hormigas_entrando_al_nido_en_cargadas': 'en_cargadas',
    'total_hormigas_entrando_al_nido_en_sin_carga': 'en_sin_carga',
    'total_hormigas_saliendo_del_nido_sn_cargadas': 'sn_cargadas',
    'total_hormigas_saliendo_del_nido_sn_sin_carga': 'sn_sin_carga',
    'velocidad_promedio_en__cargadas': 'vel_en_cargadas',
    'velocidad_promedio_en__sin_carga': 'vel_en_sin_carga',
    'velocidad_promedio_sn__cargadas': 'vel_sn_cargadas',
    'velocidad_promedio_sn__sin_carga': 'vel_sn_sin_carga',
    'rea_mediana_en__cargadas': 'area_en_cargadas',
    'rea_mediana_en__sin_carga': 'area_en_sin_carga',
    'rea_mediana_sn__cargadas': 'area_sn_cargadas',
    'rea_mediana_sn__sin_carga': 'area_sn_sin_carga',
    'largo_mediana_en__cargadas': 'largo_en_cargadas',
    'largo_mediana_en__sin_carga': 'largo_en_sin_carga',
    'largo_mediana_sn__cargadas': 'largo_sn_cargadas',
    'largo_mediana_sn__sin_carga': 'largo_sn_sin_carga',
    'ancho_mediana_en__cargadas': 'ancho_en_cargadas',
    'ancho_mediana_en__sin_carga': 'ancho_en_sin_carga',
    'ancho_mediana_sn__cargadas': 'ancho_sn_cargadas',
    'ancho_mediana_sn__sin_carga': 'ancho_sn_sin_carga',
    'temp_media': 'temp_media',
    'temp_max': 'temp_max',
    'temp_mín': 'temp_min',
    'dew point_media': 'dew_point_media',
    'dew point_mín': 'dew_point_min',
    'solar radiation dgt_media': 'rad_solar_media',
    'relative humidity_media': 'hum_rel_media',
    'relative humidity_max': 'hum_rel_max',
    'relative humidity_mín': 'hum_rel_min',
    'precip_total': 'precip_total',
    
    # --- Añadiendo columnas de scripts de EDA ---
    # (Asegúrate de que los nombres de clave coincidan EXACTAMENTE con el Excel)
    'Total hormigas entrando al nido (EN)_Cargadas': 'en_cargadas',
    'Total hormigas entrando al nido (EN)_Sin carga': 'en_sin_carga',
    'Total hormigas saliendo del nido (SN)_Cargadas': 'sn_cargadas',
    'Total hormigas saliendo del nido (SN)_Sin carga': 'sn_sin_carga',
    'air_temperature_mean': 'temp_media',
    'Área mediana EN [mm²]_Cargadas': 'area_en_cargadas',
    'Área mediana SN [mm²]_Cargadas': 'area_sn_cargadas',
    'Área mediana EN [mm²]_Sin carga': 'area_en_sin_carga',
    'Área mediana SN [mm²]_Sin carga': 'area_sn_sin_carga',
    'Velocidad promedio EN [mm/s]_Cargadas': 'vel_en_cargadas',
    'Velocidad promedio SN [mm/s]_Cargadas': 'vel_sn_cargadas',
    'Velocidad promedio EN [mm/s]_Sin carga': 'vel_en_sin_carga',
    'Velocidad promedio SN [mm/s]_Sin carga': 'vel_sn_sin_carga',
}

# Columnas para rellenar con 0 (basado en script ML)
COLS_MOVIMIENTO = [
    'vel_en_cargadas', 'vel_en_sin_carga', 'vel_sn_cargadas', 'vel_sn_sin_carga',
    'area_en_cargadas', 'area_en_sin_carga', 'area_sn_cargadas', 'area_sn_sin_carga',
    'largo_en_cargadas', 'largo_en_sin_carga', 'largo_sn_cargadas', 'largo_sn_sin_carga',
    'ancho_en_cargadas', 'ancho_en_sin_carga', 'ancho_sn_cargadas', 'ancho_sn_sin_carga'
]

# Features para el preprocesador del modelo
FEATURES_NUMERICAS_ML = [
    'temp_media',
    'rad_solar_media',
    'hum_rel_media',
    'precip_total'
]

# --- Funciones de Carga y Procesamiento de Datos ---

@st.cache_data
def load_and_process_data(folder_path):
    """
    Carga todos los archivos .xlsx de la carpeta de datos,
    los concatena, renombra columnas y aplica feature engineering.
    """
    all_data = []

    if not os.path.exists(folder_path):
        st.error(
            f"Error: El directorio '{folder_path}' no se encontró. Asegúrate de que exista en el repositorio de GitHub.")
        return pd.DataFrame()

    file_list = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]

    if not file_list:
        st.warning(f"No se encontraron archivos .xlsx o .xls en la carpeta '{folder_path}'.")
        return pd.DataFrame()

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        try:
            df = pd.read_excel(file_path)
            df['source_file'] = file_name  # Añadir origen
            all_data.append(df)
        except Exception as e:
            st.error(f"Error al leer el archivo {file_name}: {e}")

    if not all_data:
        return pd.DataFrame()

    # Concatenar todos los DataFrames
    df_raw = pd.concat(all_data, ignore_index=True)

    # 1. Renombrar columnas
    # Filtra el diccionario de renombrado para solo incluir columnas que existen en df_raw
    relevant_rename_dict = {k: v for k, v in RENAMING_DICT.items() if k in df_raw.columns}
    df_processed = df_raw.rename(columns=relevant_rename_dict)

    # 2. Feature Engineering Básico (de ambos scripts)
    try:
        # Asegurar que las columnas clave existan antes de procesar
        if 'hora_inicio' not in df_processed.columns:
            st.error("Columna 'hora_inicio' no encontrada después de renombrar. Verifica RENAMING_DICT.")
            return pd.DataFrame()

        df_processed['hora_inicio'] = pd.to_datetime(df_processed['hora_inicio'])
        df_processed = df_processed.sort_values('hora_inicio').reset_index(drop=True)

        # (Añadido para gráfico de heatmap)
        df_processed['hora'] = df_processed['hora_inicio'].dt.hour
        df_processed['dia_str'] = df_processed['hora_inicio'].dt.date.astype(str)

        # Crear totales (asegurando que las columnas existan)
        cols_en_cargadas = [c for c in ['en_cargadas'] if c in df_processed.columns]
        cols_sn_cargadas = [c for c in ['sn_cargadas'] if c in df_processed.columns]
        cols_en_sin_carga = [c for c in ['en_sin_carga'] if c in df_processed.columns]
        cols_sn_sin_carga = [c for c in ['sn_sin_carga'] if c in df_processed.columns]

        df_processed['total_entrando'] = df_processed[cols_en_cargadas].sum(axis=1, skipna=True) + df_processed[
            cols_en_sin_carga].sum(axis=1, skipna=True)
        df_processed['total_saliendo'] = df_processed[cols_sn_cargadas].sum(axis=1, skipna=True) + df_processed[
            cols_sn_sin_carga].sum(axis=1, skipna=True)
        df_processed['total_cargadas'] = df_processed[cols_en_cargadas].sum(axis=1, skipna=True) + df_processed[
            cols_sn_cargadas].sum(axis=1, skipna=True)
        df_processed['total_sin_carga'] = df_processed[cols_en_sin_carga].sum(axis=1, skipna=True) + df_processed[
            cols_sn_sin_carga].sum(axis=1, skipna=True)

        df_processed['total_hormigas'] = df_processed['total_entrando'] + df_processed['total_saliendo']

        # --- Feature Engineering para ML (MODIFICADO) ---
        # Definir los rangos y etiquetas para la clasificación multiclase
        # Bins: (min] a 0], (0] a 20], (20] a 60], (60] a 100], (100] a max]
        bins = [-float('inf'), 0, 20, 60, 100, float('inf')]
        labels = ["Nula", "Baja", "Media", "Alta", "Extrema"]

        # Usamos 'total_cargadas' como base para la actividad
        # Rellenamos NaNs en 'total_cargadas' con 0 antes de cortar
        df_processed['ActividadRango'] = pd.cut(
            df_processed['total_cargadas'].fillna(0),
            bins=bins,
            labels=labels,
            right=True  # right=True incluye el valor del 'bin' (ej. 0 se incluye en 'Nula')
        )

        # Guardar los rangos para usarlos en el reporte
        st.session_state.class_labels = labels
        # --- Fin de la Modificación ---

        # Feature Engineering para gráficos Altair
        cols_area = [c for c in ['area_en_cargadas', 'area_en_sin_carga', 'area_sn_cargadas', 'area_sn_sin_carga'] if
                     c in df_processed.columns]
        df_processed['tamano_promedio'] = df_processed[cols_area].mean(axis=1, skipna=True)

        if not df_processed.dropna(subset=['tamano_promedio']).empty:
            df_processed['categoria_tamano'] = pd.qcut(
                df_processed.dropna(subset=['tamano_promedio'])['tamano_promedio'], 3,
                labels=['Pequeñas', 'Medianas', 'Grandes'])

        return df_processed

    except Exception as e:
        st.error(f"Error durante el feature engineering: {e}")
        st.dataframe(df_processed.head())  # Muestra dónde falló
        return pd.DataFrame()


# --- Funciones de Gráficos (Adaptadas para usar nombres de columna nuevos) ---

# (Todas las funciones de Matplotlib/Seaborn han sido eliminadas:
# plot_totales_simple, plot_cargadas_vs_sin, plot_entrando_saliendo_cargadas,
# plot_correlacion_temperatura, plot_area_vs_velocidad, plot_heatmap_tamaño_hora)


# --- Funciones de Gráficos (Altair) ---

def get_altair_chart_temp_actividad(df, temp_range=None, ants_range=None):
    """
    Altair Scatter: Temperatura vs Actividad Total.
    Acepta rangos opcionales para fijar los ejes.

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    temp_range (list): Opcional. Lista con [min, max] para el eje X (temperatura).
    ants_range (list): Opcional. Lista con [min, max] para el eje Y (hormigas).
    """

    # --- 1. Diagnóstico de columnas (de nuestras correcciones anteriores) ---
    columnas_necesarias = ['temp_media', 'total_hormigas', 'fecha_hora_sensor']
    columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]

    if columnas_faltantes:
        st.error(f"Error en Gráfico Temperatura: Faltan columnas: {columnas_faltantes}")
        return None

    # --- 2. Filtro interno (como estaba antes) ---
    df_plot = df[(df['temp_media'] >= 0) & (df['temp_media'] <= 40) & (df['total_hormigas'] >= 0)]

    # --- 3. Diagnóstico de datos vacíos (de nuestras correcciones anteriores) ---
    if df_plot.empty:
        st.warning(f"No hay datos válidos (Temp >= 0 y Total >= 0) para mostrar en la selección actual.")
        return None

    # --- 4. INICIO DE LA MODIFICACIÓN (Ejes condicionales) ---

    # Definir el eje X
    if temp_range:
        # Si nos dan un rango, lo usamos en alt.Scale(domain=...)
        x_axis_encoding = alt.X('temp_media:Q',
                                title='Temperatura media (°C)',
                                scale=alt.Scale(domain=temp_range))
    else:
        # Si no, dejamos que Altair decida (auto-ajuste)
        x_axis_encoding = alt.X('temp_media:Q',
                                title='Temperatura media (°C)')

    # Definir el eje Y
    if ants_range:
        # Si nos dan un rango, lo usamos
        y_axis_encoding = alt.Y('total_hormigas:Q',
                                title='Total de hormigas',
                                scale=alt.Scale(domain=ants_range))
    else:
        # Si no, dejamos que Altair decida
        y_axis_encoding = alt.Y('total_hormigas:Q',
                                title='Total de hormigas')

    # --- FIN DE LA MODIFICACIÓN ---

    # --- 5. Construir el gráfico ---
    chart = (
        alt.Chart(df_plot)
        .mark_circle(size=80, opacity=0.55, color='#2563eb', stroke='#1e3a8a', strokeWidth=0.6)
        .encode(
            # Usamos las variables que definimos arriba
            x=x_axis_encoding,
            y=y_axis_encoding,
            tooltip=[
                alt.Tooltip('fecha_hora_sensor:T', title='Fecha y hora'),
                alt.Tooltip('temp_media:Q', title='Temperatura (°C)', format=".1f"),
                alt.Tooltip('total_hormigas:Q', title='Hormigas totales', format=",.0f")
            ]
        )
        .properties(
            title={
                "text": "Actividad de Hormigas según Temperatura Media",
                "subtitle": "Cada punto representa una observación individual"
            }
        )
        .interactive()
    )
    return chart


def get_altair_heatmap_hora_tamano(df):
    """Altair Heatmap: Actividad por Hora y Tamaño."""
    if 'categoria_tamano' not in df.columns or 'hora' not in df.columns or 'total_hormigas' not in df.columns:
        return None

    # --- CAMBIO 1: De .sum() a .mean() ---
    agrupado = (
        df.groupby(['hora', 'categoria_tamano'], as_index=False, observed=True)
        .agg({'total_hormigas': 'mean'}) # <-- CAMBIADO DE 'sum' A 'mean'
    )

    chart = (
        alt.Chart(agrupado)
        .mark_rect(strokeWidth=0)
        .encode(
            x=alt.X('hora:O', title='Hora del día', sort=list(range(24)), axis=alt.Axis(labelAngle=0)),
            y=alt.Y('categoria_tamano:N', title='Tamaño de hormigas', sort=['Pequeñas', 'Medianas', 'Grandes']),
            color=alt.Color(
                'total_hormigas:Q',
                # --- CAMBIO 2: Título y escala logarítmica ---
                title='Actividad promedio (media)', # <-- Título actualizado
                scale=alt.Scale(scheme='goldred', type='log') # <-- AÑADIDO type='log'
            ),
            tooltip=[
                alt.Tooltip('hora:O', title='Hora'),
                alt.Tooltip('categoria_tamano:N', title='Tamaño'),
                # --- CAMBIO 3: Tooltip actualizado ---
                alt.Tooltip('total_hormigas:Q', title='Actividad promedio', format=',.1f') # <-- Título y formato
            ]
        )
        .properties(
            title={
                "text": "Mapa de Calor de Actividad PROMEDIO de Hormigas por Hora", # <-- Título actualizado
                "subtitle": "Promedio de entradas y salidas agrupado por hora y tamaño corporal"
            }
        )
    )
    return chart

def get_altair_heatmap_temp_tamano(df):
    """Altair Heatmap: Actividad por Temperatura y Tamaño."""
    if 'categoria_tamano' not in df.columns or 'temp_media' not in df.columns or 'total_hormigas' not in df.columns:
        return None

    df_plot = df.dropna(subset=['tamano_promedio', 'temp_media'])
    df_plot['temp_bin'] = pd.cut(df_plot['temp_media'], bins=15)

    # --- CAMBIO 1: De .sum() a .mean() ---
    agrupado_temp = (
        df_plot.groupby(['temp_bin', 'categoria_tamano'], as_index=False, observed=True)
        .agg({'total_hormigas': 'mean'}) # <-- CAMBIADO DE 'sum' A 'mean'
    )

    agrupado_temp['temp_label'] = agrupado_temp['temp_bin'].apply(lambda x: f"{x.left:.1f}–{x.right:.1f}°C")
    agrupado_temp['temp_sort_key'] = agrupado_temp['temp_bin'].apply(lambda x: x.left)

    chart_temp = (
        alt.Chart(agrupado_temp)
        .mark_rect(strokeWidth=0)
        .encode(
            x=alt.X(
                'temp_label:O',
                title='Temperatura media (°C)',
                axis=alt.Axis(labelAngle=45),
                sort=alt.SortField('temp_sort_key')
            ),
            y=alt.Y('categoria_tamano:N', title='Tamaño de hormigas', sort=['Pequeñas', 'Medianas', 'Grandes']),
            color=alt.Color(
                'total_hormigas:Q',
                # --- CAMBIO 2: Título y escala logarítmica ---
                title='Actividad promedio (media)', # <-- Título actualizado
                scale=alt.Scale(scheme='goldred', type='log') # <-- AÑADIDO type='log'
            ),
            tooltip=[
                alt.Tooltip('temp_label:O', title='Temperatura (°C)'),
                alt.Tooltip('categoria_tamano:N', title='Tamaño'),
                # --- CAMBIO 3: Tooltip actualizado ---
                alt.Tooltip('total_hormigas:Q', title='Actividad promedio', format=',.1f') # <-- Título y formato
            ]
        )
        .properties(
            title={
                "text": "Mapa de Calor de Actividad PROMEDIO de Hormigas según Temperatura", # <-- Título actualizado
                "subtitle": "Promedio de entradas y salidas agrupado por temperatura y tamaño corporal"
            }
        )
    )
    return chart_temp
    
def get_altair_boxplot_velocidad(df):
    """Altair Boxplot: Comparación de Velocidad por Carga."""
    vel_cols = [c for c in df.columns if c in [
        'vel_en_cargadas', 'vel_en_sin_carga', 
        'vel_sn_cargadas', 'vel_sn_sin_carga'
    ]]
    
    if not vel_cols:
        return None

    df_melt = (
        df.melt(value_vars=vel_cols, var_name='tipo', value_name='velocidad')
        .dropna(subset=['velocidad'])
    )

    df_melt['carga'] = df_melt['tipo'].apply(
        lambda x: 'Con carga' if 'cargadas' in x else 'Sin carga'
    )
    
    chart_vel = (
        alt.Chart(df_melt)
        .mark_boxplot(size=80, median={'color': 'black'})
        .encode(
            y=alt.Y('carga:N',
                    title='Tipo de carga',
                    sort=['Con carga', 'Sin carga']),
            x=alt.X('velocidad:Q',
                    title='Velocidad promedio (mm/s)'),
            color=alt.Color(
                'carga:N',
                scale=alt.Scale(domain=['Con carga', 'Sin carga'],
                                range=['#d73027', '#fee08b']),
                legend=None
            ),
            tooltip=[
                alt.Tooltip('carga:N', title='Tipo de carga'),
                alt.Tooltip('velocidad:Q', title='Velocidad promedio', format='.2f')
            ]
        )
        .properties(
            title={
                "text": "Comparación de Velocidad de Hormigas según Carga",
                "subtitle": "Distribución de velocidades combinando entrada y salida del nido"
            },
            height=400
        )
    )
    return chart_vel


def get_altair_scatter_temp_tamano(df, temp_range=None, size_range=None):
    """
    Altair Scatter: Temperatura vs Tamaño.
    Acepta rangos opcionales para fijar los ejes.

    Parámetros:
    df (pd.DataFrame): DataFrame con los datos.
    temp_range (list): Opcional. Lista con [min, max] para el eje X (temperatura).
    size_range (list): Opcional. Lista con [min, max] para el eje Y (tamaño).
    """

    # --- 1. Diagnóstico de columnas ---
    columnas_necesarias = ['temp_media', 'tamano_promedio', 'categoria_tamano']
    columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]

    if columnas_faltantes:
        st.error(f"Error en Gráfico Tamaño: Faltan columnas: {columnas_faltantes}")
        return None

    # --- 2. Filtro interno (eliminar filas sin datos clave) ---
    df_plot = df.dropna(subset=['temp_media', 'tamano_promedio', 'categoria_tamano'])

    # --- 3. Diagnóstico de datos vacíos ---
    if df_plot.empty:
        st.warning(f"No hay datos válidos (con Temp, Tamaño y Categoría) para mostrar en la selección actual.")
        return None

    # --- 4. Definir eje X (Temperatura) ---
    if temp_range:
        x_axis_encoding = alt.X('temp_media:Q',
                                title='Temperatura media (°C)',
                                scale=alt.Scale(domain=temp_range))
    else:
        x_axis_encoding = alt.X('temp_media:Q',
                                title='Temperatura media (°C)')

    # --- 5. Definir eje Y (Tamaño) ---
    if size_range:
        y_axis_encoding = alt.Y('tamano_promedio:Q',
                                title='Tamaño corporal promedio (área mm²)',
                                scale=alt.Scale(domain=size_range))
    else:
        y_axis_encoding = alt.Y('tamano_promedio:Q',
                                title='Tamaño corporal promedio (área mm²)')

    # --- 6. Construir el gráfico ---
    chart_scatter = (
        alt.Chart(df_plot)
        .mark_circle(opacity=0.4, size=60)
        .encode(
            x=x_axis_encoding,
            y=y_axis_encoding,
            color=alt.Color(
                'categoria_tamano:N',
                title='Grupo de tamaño',
                scale=alt.Scale(
                    domain=['Pequeñas', 'Medianas', 'Grandes'],
                    range=['#fee08b', '#f46d43', '#d73027']
                )
            ),
            tooltip=[
                alt.Tooltip('temp_media:Q', title='Temperatura (°C)', format='.1f'),
                alt.Tooltip('tamano_promedio:Q', title='Tamaño promedio', format='.2f'),
                alt.Tooltip('categoria_tamano:N', title='Grupo')
            ]
        )
        .properties(
            title={
                "text": "Relación entre Temperatura y Tamaño Corporal"
            }
        )
        .interactive()
    )
    return chart_scatter

# --- Funciones de Machine Learning ---

class DropColumns(BaseEstimator, TransformerMixin):
    """Clase para eliminar columnas en un Pipeline."""
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors="ignore")


@st.cache_resource
def train_models(df):
    """
    Entrena el modelo SVM multiclase y devuelve el pipeline, métricas, y X_train.
    USA UNA DIVISIÓN DE TIEMPO 80/20 (ENTRENAR CON EL PASADO, PROBAR CON EL FUTURO)
    """

    # --- INICIO DE LA CORRECCIÓN DE ALINEACIÓN ---
    # 1. Preparar datos para ML

    # Primero, dropear NaNs de las features que SÍ O SÍ necesitamos
    df_cleaned = df.dropna(subset=FEATURES_NUMERICAS_ML + ['ActividadRango']).copy()

    # Crear y alinear X e y DESPUÉS de limpiar
    y = df_cleaned['ActividadRango'].shift(-1)
    X = df_cleaned.copy()

    # Recortar X e y para alinear (shift)
    X = X.iloc[2:-1].reset_index(drop=True)
    y = y.iloc[2:-1].reset_index(drop=True)

    # Volver a alinear X e y por si el shift generó NaNs al final
    valid_indices = y.dropna().index
    X = X.loc[valid_indices].reset_index(drop=True)
    y = y.loc[valid_indices].reset_index(drop=True)

    # Ahora X e y están 100% alineados y limpios
    # --- FIN DE LA CORRECCIÓN DE ALINEACIÓN ---

    # Forzar a 'y' a ser un tipo Categórico CON TODAS las clases.
    class_labels = st.session_state.class_labels
    y = y.astype(pd.CategoricalDtype(categories=class_labels))

    # 2. Separación Train/Test temporal (80% train, 20% test)
    dias_unicos = X['dia_str'].unique()
    dias_unicos.sort()  # ¡Muy importante ordenar los días!

    if len(dias_unicos) < 2:
        st.error("Error de ML: Se necesitan al menos 2 días de datos para dividir en train/test.")
        return None, None, None, None  # Añadido un None extra

    split_point = int(len(dias_unicos) * 0.8)

    if split_point == len(dias_unicos):
        split_point -= 1

    DIAS_DE_TRAIN = dias_unicos[:split_point]
    DIAS_DE_TEST = dias_unicos[split_point:]

    dias_test_str = ", ".join(DIAS_DE_TEST) if len(
        DIAS_DE_TEST) < 4 else f"{len(DIAS_DE_TEST)} días (desde {DIAS_DE_TEST[0]})"

    train_mask = X['dia_str'].isin(DIAS_DE_TRAIN)
    test_mask = X['dia_str'].isin(DIAS_DE_TEST)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if X_train.empty or X_test.empty:
        st.error(
            f"Error de ML: La división de tiempo resultó en un set vacío. Train: {len(X_train)}, Test: {len(X_test)}")
        return None, None, None, None

    st.session_state.ml_info = {
        "dia_test": dias_test_str,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    # 3. Pipeline de Preprocesamiento (Queda igual)
    cols_to_drop = [col for col in X.columns if col not in FEATURES_NUMERICAS_ML]
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    preprocessor = Pipeline([
        ("drop_cols", DropColumns(columns_to_drop=cols_to_drop)),
        ("column_transformer", ColumnTransformer(
            [("num", numeric_pipeline, FEATURES_NUMERICAS_ML)],
            remainder="drop"
        ))
    ])

    # 4. Definición y Entrenamiento de SVM (Queda igual)
    modelo_svm = SVC(C=0.3058, gamma="auto", kernel="rbf",
                     class_weight="balanced", probability=True, random_state=42)

    pipeline_final = Pipeline([
        ("preprocessing", preprocessor),
        ("model", modelo_svm)
    ])

    pipeline_final.fit(X_train, y_train)
    preds = pipeline_final.predict(X_test)

    # 5. Métricas Multiclase (Queda igual)
    report_dict = classification_report(
        y_test, preds, labels=class_labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, preds, labels=class_labels)
    f1_w = f1_score(y_test, preds, average='weighted')

    resultados_metricas = {
        "report_df": pd.DataFrame(report_dict).transpose(),
        "cm": cm,
        "f1_weighted": f1_w
    }

    return pipeline_final, resultados_metricas, X_train, y_train


@st.cache_resource
def get_shap_explainer(_pipeline, _X_train, _y_train_ignored):  # Renombramos y_train para no usarlo
    """
    Crea y cachea un explicador SHAP (KernelExplainer) para el modelo SVM.
    Usa el PROPIO MODELO para predecir las etiquetas de X_train y crear un resumen balanceado.
    """
    st.info("Preparando el explicador SHAP (esto ocurre solo una vez)...")

    # 1. Obtener el preprocesador y el modelo
    preprocessor = _pipeline.named_steps['preprocessing']
    model = _pipeline.named_steps['model']

    # 2. Transformar los datos de entrenamiento (background data)
    X_train_transformed = preprocessor.transform(_X_train)

    # --- INICIO DE LA CORRECCIÓN: Resumen Balanceado ---

    # 3. PREDECIR las etiquetas de los datos de entrenamiento transformados
    # Esta es la corrección clave. No usamos y_train (desbalanceado),
    # usamos lo que el modelo PREDICE sobre X_train.
    train_labels = model.predict(X_train_transformed)

    # Convertir X_train_transformed a DataFrame para filtrar por label
    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=FEATURES_NUMERICAS_ML)
    X_train_transformed_df['label'] = train_labels  # Usar las etiquetas predichas

    summary_list = []
    N_SAMPLES_PER_CLASS = 25  # Muestras por clase para el resumen

    # Iterar sobre las clases que el modelo CONOCE
    for label in _pipeline.classes_:

        class_data = X_train_transformed_df[X_train_transformed_df['label'] == label]
        n_samples = min(len(class_data), N_SAMPLES_PER_CLASS)

        if n_samples > 0:
            summary_list.append(class_data.drop(columns='label').sample(n_samples, random_state=42))

    if not summary_list:
        st.warning("El resumen de SHAP no pudo balancearse, usando K-Means.")
        X_train_summary = shap.kmeans(X_train_transformed, 50)
    else:
        X_train_summary = pd.concat(summary_list).values
        st.success(
            f"Explicador SHAP creado con un resumen balanceado de {len(X_train_summary)} puntos de {len(summary_list)} clases.")

    # --- FIN DE LA CORRECCIÓN ---

    # 4. Crear la función de predicción
    def predict_fn(x):
        return model.predict_proba(x)

    # 5. Crear el explicador
    explainer = shap.KernelExplainer(predict_fn, X_train_summary)

    return explainer


# Cargar y procesar datos al inicio
df_processed = load_and_process_data(DATA_FOLDER)

# --- Barra Lateral (Navegación) ---
st.sidebar.title("🐜 App de Análisis de Hormigas")
st.sidebar.markdown("---")

if df_processed.empty:
    st.sidebar.error("No se pudieron cargar datos. La aplicación está en modo limitado.")
    pagina = st.sidebar.radio("Navegación", ["Inicio"])
else:
     pagina = st.sidebar.radio("Navegación", ["Inicio", "Exploración de Datos (EDA)", "Modelo Predictivo (ML)"])

st.sidebar.markdown("---")
st.sidebar.markdown("Proyecto final de análisis de datos.")


# --- Página de Inicio ---
if pagina == "Inicio":
    st.title("🐜 Análisis de Actividad y Modelo Predictivo")
    st.markdown("""
    Bienvenido a la aplicación de análisis de actividad de hormigas. Esta herramienta es la culminación
    de un proyecto de ciencia de datos, diseñada para explorar patrones y predecir la actividad
    futura basada en condiciones ambientales.
    
    ### Objetivos de la Aplicación
    
    1.  **Explorar Datos:** Visualizar la dinámica histórica de entrada/salida de hormigas, 
        su comportamiento con/sin carga y la influencia de factores como la temperatura.
    2.  **Probar el Modelo:** Interactuar con un modelo de Machine Learning entrenado para 
        predecir si la actividad será "Alta" o "Baja" en el minuto siguiente.
    3.  **Validar Nuevos Datos:** Proveer una interfaz para cargar y validar nuevos
        conjuntos de datos.
        
    ### Cómo Navegar
    
    Usa el menú en la barra lateral izquierda para moverte entre las secciones:
    
    * **Cargar Nuevo Archivo:** Valida y simula la carga de nuevos archivos de datos.
    * **Exploración de Datos (EDA):** Contiene todos los gráficos descriptivos.
    * **Modelo Predictivo (ML):** Muestra el rendimiento del modelo y te permite
        hacer predicciones en tiempo real.
    """)
    
    if not df_processed.empty:
        st.subheader("Resumen de Datos Cargados")
        n_archivos = len(df_processed['source_file'].unique())
        n_registros = len(df_processed)
        fecha_inicio = df_processed['hora_inicio'].min().strftime('%Y-%m-%d')
        fecha_fin = df_processed['hora_inicio'].max().strftime('%Y-%m-%d')
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Archivos Cargados", n_archivos)
        col2.metric("Total Registros", f"{n_registros:,}")
        col3.metric("Desde", fecha_inicio)
        col4.metric("Hasta", fecha_fin)
        
        st.dataframe(df_processed.sample(5))
    else:
        st.error("No se han podido cargar los datos de la carpeta `datos/`. Por favor, revisa las instrucciones en `README.md`.")


# --- Página de Exploración de Datos (EDA) ---
elif pagina == "Exploración de Datos (EDA)":
    st.title("📈 Exploración de Datos (EDA)")
    st.markdown(
        "Visualización de los patrones de actividad de las hormigas basados en todos los datos históricos cargados.")

    # Asegurarse de que los datos estén cargados
    if df_processed.empty:
        st.error("No se pueden mostrar gráficos porque no se cargaron datos.")
    else:
        st.subheader("Análisis de Distribuciones (Altair)")
        st.markdown("Gráficos interactivos para explorar relaciones entre variables.")

        # --- Gráfico 1: Temperatura vs. Actividad Total (CON FILTRO) ---

        # 1. Calcular rangos globales ANTES de filtrar
        global_temp_min = df_processed['temp_media'].min()
        global_temp_max = df_processed['temp_media'].max()
        global_ants_min = df_processed['total_hormigas'].min()
        global_ants_max = df_processed['total_hormigas'].max()

        temp_domain = [global_temp_min, global_temp_max]
        ants_domain = [global_ants_min, global_ants_max]

        # 2. DataFrame por defecto (todos los archivos)
        df_to_plot_chart1 = df_processed.copy()
        selected_file_label = "Todos los archivos"  # Etiqueta para el spinner

        # 3. Lógica del filtro (¡CORREGIDA!)
        if 'source_file' in df_processed.columns:
            file_list = sorted(df_processed['source_file'].unique())
            options = ["Todos los archivos"] + file_list

            selected_file = st.selectbox(
                "Filtrar por archivo (Gráfico de Temperatura vs. Actividad):",
                options=options,
                index=0
            )

            # Si se selecciona un archivo específico, filtrar
            if selected_file != "Todos los archivos":
                df_to_plot_chart1 = df_processed[df_processed['source_file'] == selected_file]
                selected_file_label = selected_file  # Actualizar etiqueta

        else:
            st.warning("No se encontró la columna 'source_file' para el filtro.")

        # 4. Generar el gráfico (¡CON INDENTACIÓN CORREGIDA!)
        # Esto ahora está FUERA del 'if/else' del filtro,
        # así que se ejecuta siempre, con los datos correctos.
        with st.spinner(f"Generando Scatter: Temperatura vs. Actividad (Archivo: {selected_file_label})..."):
            chart1 = get_altair_chart_temp_actividad(
                df_to_plot_chart1,
                temp_range=temp_domain,
                ants_range=ants_domain
            )

            if chart1:
                st.altair_chart(chart1, use_container_width=True)
            # (El 'st.warning' de "No hay datos válidos..." ya está dentro de la función)

        st.markdown("---")

        # --- Gráfico 2: Heatmap Temperatura y Tamaño ---
        with st.spinner("Generando Heatmap: Actividad por Temperatura y Tamaño..."):
            chart2 = get_altair_heatmap_temp_tamano(df_processed)
            if chart2:
                st.altair_chart(chart2, use_container_width=True)
            else:
                st.warning(
                    "No se pudo generar 'Heatmap: Actividad por Temperatura y Tamaño'. Verifica las columnas 'categoria_tamano'.")

        st.markdown("---")

        # --- Gráfico 3: Scatter Temperatura vs. Tamaño Corporal (CON FILTRO) ---

        # 1. Calcular rangos globales (X=temp, Y=size)
        # (Aseguramos que no fallen si hay NaNs)
        g3_temp_min = df_processed['temp_media'].min()
        g3_temp_max = df_processed['temp_media'].max()
        g3_size_min = df_processed['tamano_promedio'].min()
        g3_size_max = df_processed['tamano_promedio'].max()

        # Guardar en listas
        g3_temp_domain = [g3_temp_min, g3_temp_max]
        g3_size_domain = [g3_size_min, g3_size_max]

        # 2. DataFrame por defecto (todos los archivos)
        df_to_plot_chart3 = df_processed.copy()
        selected_file_label_g3 = "Todos los archivos"

        # 3. Lógica del filtro
        if 'source_file' in df_processed.columns:
            file_list_g3 = sorted(df_processed['source_file'].unique())
            options_g3 = ["Todos los archivos"] + file_list_g3

            selected_file_g3 = st.selectbox(
                "Filtrar por archivo (Gráfico de Temperatura vs. Tamaño):",
                options=options_g3,
                index=0,
                key='selectbox_chart_3'  # Clave única para este selectbox
            )

            if selected_file_g3 != "Todos los archivos":
                df_to_plot_chart3 = df_processed[df_processed['source_file'] == selected_file_g3]
                selected_file_label_g3 = selected_file_g3

        else:
            st.warning("No se encontró la columna 'source_file' para el filtro (Gráfico 3).")

        # 4. Generar el gráfico
        with st.spinner(f"Generando Scatter: Temperatura vs. Tamaño (Archivo: {selected_file_label_g3})..."):
            chart3 = get_altair_scatter_temp_tamano(
                df_to_plot_chart3,
                temp_range=g3_temp_domain,
                size_range=g3_size_domain
            )

            if chart3:
                st.altair_chart(chart3, use_container_width=True)
        st.markdown("---")

        # --- Gráfico 4: Heatmap Hora y Tamaño ---
        with st.spinner("Generando Heatmap: Actividad por Hora y Tamaño..."):
            chart4 = get_altair_heatmap_hora_tamano(df_processed)
            if chart4:
                st.altair_chart(chart4, use_container_width=True)
            else:
                st.warning(
                    "No se pudo generar 'Heatmap: Actividad por Hora y Tamaño'. Verifica las columnas 'categoria_tamano'.")

        st.markdown("---")

        # --- Gráfico 5: Boxplot Velocidad por Carga ---
        with st.spinner("Generando Boxplot: Velocidad por Tipo de Carga..."):
            chart5 = get_altair_boxplot_velocidad(df_processed)
            if chart5:
                st.altair_chart(chart5, use_container_width=True)
            else:
                st.warning("No se pudo generar 'Boxplot: Velocidad por Tipo de Carga'.")


# --- Página de Modelo Predictivo (ML) ---
elif pagina == "Modelo Predictivo (ML)":

    # Inicializar SHAP
    shap.initjs()

    st.title("🤖 Modelo Predictivo (ML)")

    if df_processed.empty:
        st.error("No se pueden entrenar modelos porque no se cargaron datos.")
    else:
        with st.spinner("Entrenando modelo SVM... (Esto puede tardar un momento la primera vez)"):
            pipeline_prediccion, metricas, X_train_data, y_train_data = train_models(df_processed)

        if pipeline_prediccion is None:
            st.error("Falló el entrenamiento del modelo. Revisa los mensajes de error anteriores.")
        else:
            explainer_shap = get_shap_explainer(pipeline_prediccion, X_train_data, y_train_data)

            ml_info = st.session_state.get('ml_info', {})

            # Usar las clases que el modelo REALMENTE aprendió
            model_classes = pipeline_prediccion.classes_
            st.session_state.model_classes = model_classes  # Guardar para usar después

            st.subheader("Prueba de Predicción y Explicación (SHAP)")

            st.info("""
            **Rangos de Actividad (hormigas cargadas por minuto):**
            - **Nula:** 0
            - **Baja:** 1 - 20
            - **Media:** 21 - 60
            - **Alta:** 61 - 100
            - **Extrema:** 101 o más
            """)

            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                # ... (Sliders de formulario - sin cambios)
                with col1:
                    temp_min = float(df_processed['temp_media'].min())
                    temp_max = float(df_processed['temp_media'].max())
                    temp_val = float(df_processed['temp_media'].mean())
                    input_temp = st.slider("🌡️ Temperatura Media (°C)", temp_min, temp_max, temp_val)
                    hum_min = float(df_processed['hum_rel_media'].min())
                    hum_max = float(df_processed['hum_rel_media'].max())
                    hum_val = float(df_processed['hum_rel_media'].mean())
                    input_hum = st.slider("💧 Humedad Relativa (%)", hum_min, hum_max, hum_val)
                with col2:
                    rad_min = float(df_processed['rad_solar_media'].min())
                    rad_max = float(df_processed['rad_solar_media'].max())
                    rad_val = float(df_processed['rad_solar_media'].mean())
                    input_rad = st.slider("☀️ Radiación Solar (W/m²)", rad_min, rad_max, rad_val)
                    input_precip = st.number_input("🌧️ Precipitación Total (mm)", min_value=0.0, max_value=50.0,
                                                   value=0.0, step=0.1)

                submitted = st.form_submit_button("Predecir y Explicar")

            if submitted:
                # 1. Crear DataFrame para predicción
                input_data = pd.DataFrame({
                    'temp_media': [input_temp],
                    'rad_solar_media': [input_rad],
                    'hum_rel_media': [input_hum],
                    'precip_total': [input_precip]
                })
                for col in X_train_data.columns:
                    if col not in input_data.columns:
                        input_data[col] = 0

                try:
                    # Obtener las clases que el modelo aprendió
                    model_classes = st.session_state.model_classes

                    # 2. Predecir probabilidades
                    probabilidades = pipeline_prediccion.predict_proba(input_data)[0]

                    # 3. Obtener la predicción desde la prob. más alta
                    predicted_class_index = np.argmax(probabilidades)
                    prediccion = model_classes[predicted_class_index]
                    confianza_predicha = probabilidades[predicted_class_index]
                    prob_dict = {label: prob for label, prob in zip(model_classes, probabilidades)}

                    st.markdown("---")
                    st.subheader("Resultado de la Predicción:")

                    if prediccion == "Extrema" or prediccion == "Alta":
                        emoji = "🐜"
                    elif prediccion == "Media":
                        emoji = "🚶"
                    else:
                        emoji = "📉"

                    st.metric("Nivel de Actividad", f"{emoji} {prediccion}",
                              f"{confianza_predicha * 100:.1f}% de confianza")

                    st.subheader("Resumen de la Predicción (Explicación SHAP)")
                    st.markdown("""
                    Este gráfico muestra qué variables (features) influyeron en la predicción.
                    - Las flechas **rojas** (como ☀️) empujan la predicción hacia un nivel de actividad **más alto**.
                    - Las flechas **azules** (como 💧) empujan la predicción hacia un nivel **más bajo**.
                    """)

                    # 4. Calcular valores SHAP
                    preprocessor = pipeline_prediccion.named_steps['preprocessing']
                    input_data_transformed = preprocessor.transform(input_data)  # Shape (1, 4)

                    shap_values = explainer_shap.shap_values(input_data_transformed)

                    # 'expected_value' tiene 5 valores, uno por clase. Tomamos el de la clase predicha.
                    expected_value_class = explainer_shap.expected_value[predicted_class_index]

                    # --- INICIO DE LA CORRECCIÓN FINAL ---

                    # shap_values es un ndarray (1, 4, 5)
                    # Tomamos el primer sample [0]
                    # Tomamos TODAS las features [:]
                    # Tomamos la clase que predijimos [predicted_class_index]
                    # El resultado, 'shap_values_class', es un array (4,)

                    if isinstance(shap_values, list):
                        # Si es una lista (debería serlo), tomar el índice y la fila
                        shap_values_class = shap_values[predicted_class_index][0]
                    else:
                        # Si es un ndarray (1, 4, 5), cortarlo
                        st.warning("SHAP no devolvió una lista de clases. La explicación puede ser incorrecta.")
                        shap_values_class = shap_values[0, :, predicted_class_index]

                    # 'features' debe ser un array 1D (4,), no 2D (1, 4)
                    features_1d = input_data_transformed[0]

                    # --- FIN DE LA CORRECCIÓN FINAL ---

                    # Crear el gráfico de fuerza
                    shap_force_plot = shap.force_plot(
                        expected_value_class,
                        shap_values_class.astype(float),  # <-- FORZAR A FLOAT
                        features=features_1d.astype(float),  # <-- FORZAR A FLOAT
                        feature_names=FEATURES_NUMERICAS_ML
                    )

                    st.components.v1.html(shap_force_plot.data, height=150)

                    with st.expander("Ver detalles de la predicción"):
                        st.write("Probabilidades (solo de clases que el modelo aprendió):")
                        st.dataframe(pd.Series(prob_dict, name="Probabilidad").sort_values(ascending=False))
                        st.write(f"Valores SHAP (datos brutos) para la clase '{prediccion}':")
                        st.dataframe(pd.DataFrame(shap_values_class,
                                                  index=FEATURES_NUMERICAS_ML,
                                                  columns=["Valor SHAP"]))

                except Exception as e:
                    st.error(f"Error durante la predicción o explicación SHAP: {e}")
                    st.dataframe(input_data)