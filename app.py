import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import re
import altair as alt
import warnings
from io import BytesIO

# --- LIME y Matplotlib ---
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
# Importamos TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit

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

# Diccionario para renombrar columnas.
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

# --- MODIFICADO: Se eliminó 'precip_total' ---
FEATURES_NUMERICAS_ML = [
    'temp_media',
    'rad_solar_media',
    'hum_rel_media'
]


# --- Funciones de Carga y Procesamiento de Datos ---

@st.cache_data
def load_and_process_data(folder_path):
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
            df['source_file'] = file_name
            all_data.append(df)
        except Exception as e:
            st.error(f"Error al leer el archivo {file_name}: {e}")

    if not all_data:
        return pd.DataFrame()

    df_raw = pd.concat(all_data, ignore_index=True)
    relevant_rename_dict = {k: v for k, v in RENAMING_DICT.items() if k in df_raw.columns}
    df_processed = df_raw.rename(columns=relevant_rename_dict)

    try:
        if 'hora_inicio' not in df_processed.columns:
            st.error("Columna 'hora_inicio' no encontrada después de renombrar. Verifica RENAMING_DICT.")
            return pd.DataFrame()

        df_processed['hora_inicio'] = pd.to_datetime(df_processed['hora_inicio'])
        df_processed = df_processed.sort_values('hora_inicio').reset_index(drop=True)

        df_processed['hora'] = df_processed['hora_inicio'].dt.hour
        df_processed['dia_str'] = df_processed['hora_inicio'].dt.date.astype(str)

        cols_en_cargadas = [c for c in ['en_cargadas'] if c in df_processed.columns]
        cols_sn_cargadas = [c for c in ['sn_cargadas'] if c in df_processed.columns]
        cols_en_sin_carga = [c for c in ['en_sin_carga'] if c in df_processed.columns]
        cols_sn_sin_carga = [c for c in ['sn_sin_carga'] if c in df_processed.columns]

        df_processed['total_entrando'] = df_processed[cols_en_cargadas].sum(axis=1, skipna=True) + df_processed[
            cols_en_sin_carga].sum(axis=1, skipna=True)
        df_processed['total_saliendo'] = df_processed[cols_sn_cargadas].sum(axis=1, skipna=True) + df_processed[
            cols_sn_sin_carga].sum(axis=1, skipna=True)

        # Cálculo original (se sobrescribirá en ML con la nueva lógica si es necesario)
        df_processed['total_cargadas'] = df_processed[cols_en_cargadas].sum(axis=1, skipna=True) + df_processed[
            cols_sn_cargadas].sum(axis=1, skipna=True)

        df_processed['total_sin_carga'] = df_processed[cols_en_sin_carga].sum(axis=1, skipna=True) + df_processed[
            cols_sn_sin_carga].sum(axis=1, skipna=True)
        df_processed['total_hormigas'] = df_processed['total_entrando'] + df_processed['total_saliendo']

        # Definición inicial (se usa en EDA, pero ML usará su propia definición actualizada)
        bins = [-float('inf'), 0, 20, 60, 100, float('inf')]
        labels = ["Nula", "Baja", "Media", "Alta", "Extrema"]
        df_processed['ActividadRango'] = pd.cut(
            df_processed['total_cargadas'].fillna(0),
            bins=bins,
            labels=labels,
            right=True
        )

        # Inicializamos etiquetas por defecto (ML las actualizará)
        st.session_state.class_labels = labels

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
        st.dataframe(df_processed.head())
        return pd.DataFrame()


# --- Funciones de Gráficos (Altair) ---

def get_altair_scatter_variable_actividad(df, x_col, x_title, x_range=None, ants_range=None):
    """
    Genera un scatter plot de Actividad vs Variable X elegida (Temp, Humedad, Radiación).
    """
    columnas_necesarias = [x_col, 'total_hormigas', 'fecha_hora_sensor']
    # Verificar si existen las columnas
    if not all(col in df.columns for col in columnas_necesarias):
        st.error(f"Error en Gráfico: Faltan columnas: {[c for c in columnas_necesarias if c not in df.columns]}")
        return None

    # Filtrar datos válidos
    df_plot = df.dropna(subset=[x_col, 'total_hormigas'])

    if df_plot.empty:
        st.warning(f"No hay datos válidos para mostrar en la selección actual.")
        return None

    # Configurar escalas
    if x_range:
        x_axis_encoding = alt.X(f'{x_col}:Q', title=x_title, scale=alt.Scale(domain=x_range))
    else:
        x_axis_encoding = alt.X(f'{x_col}:Q', title=x_title)

    if ants_range:
        y_axis_encoding = alt.Y('total_hormigas:Q', title='Total de hormigas', scale=alt.Scale(domain=ants_range))
    else:
        y_axis_encoding = alt.Y('total_hormigas:Q', title='Total de hormigas')

    chart = (
        alt.Chart(df_plot)
        .mark_circle(size=80, opacity=0.55, color='#2563eb', stroke='#1e3a8a', strokeWidth=0.6)
        .encode(
            x=x_axis_encoding,
            y=y_axis_encoding,
            tooltip=[
                alt.Tooltip('fecha_hora_sensor:T', title='Fecha y hora'),
                alt.Tooltip(f'{x_col}:Q', title=x_title, format=".1f"),
                alt.Tooltip('total_hormigas:Q', title='Hormigas totales', format=",.0f")
            ]
        )
        .properties(
            title={"text": f"Actividad de Hormigas según {x_title}",
                   "subtitle": "Cada punto representa una observación individual"}
        )
        .interactive()
    )
    return chart


def get_altair_line_hora_tamano(df):
    """Reemplaza el heatmap por un gráfico de líneas (Promedio Actividad vs Hora, colores por Tamaño)"""
    if 'categoria_tamano' not in df.columns or 'hora' not in df.columns or 'total_hormigas' not in df.columns:
        return None

    agrupado = (
        df.groupby(['hora', 'categoria_tamano'], as_index=False, observed=True)
        .agg({'total_hormigas': 'mean'})
    )

    chart = (
        alt.Chart(agrupado)
        .mark_line(point=True)  # Añadir puntos para ver mejor los datos
        .encode(
            x=alt.X('hora:Q', title='Hora del día', scale=alt.Scale(domain=[0, 23])),
            y=alt.Y('total_hormigas:Q', title='Actividad Promedio (Hormigas)'),
            color=alt.Color('categoria_tamano:N', title='Tamaño de Hormiga',
                            scale=alt.Scale(scheme='category10'),
                            legend=alt.Legend(orient='top')),
            tooltip=[
                alt.Tooltip('hora:Q', title='Hora'),
                alt.Tooltip('categoria_tamano:N', title='Tamaño'),
                alt.Tooltip('total_hormigas:Q', title='Actividad Promedio', format=',.1f')
            ]
        )
        .properties(
            title={"text": "Evolución de Actividad Promedio por Hora",
                   "subtitle": "Comparación entre tamaños de hormigas (media de actividad)"}
        )
        .interactive()
    )
    return chart


def get_altair_line_temp_tamano(df):
    """Reemplaza el heatmap por un gráfico de líneas (Promedio Actividad vs Temperatura, colores por Tamaño)"""
    if 'categoria_tamano' not in df.columns or 'temp_media' not in df.columns or 'total_hormigas' not in df.columns:
        return None

    df_plot = df.dropna(subset=['tamano_promedio', 'temp_media', 'categoria_tamano']).copy()

    # Crear bins y calcular el punto medio para tener un eje X numérico continuo
    df_plot['temp_bin'] = pd.cut(df_plot['temp_media'], bins=15)
    df_plot['temp_mid'] = df_plot['temp_bin'].apply(lambda x: x.mid)

    agrupado_temp = (
        df_plot.groupby(['temp_mid', 'categoria_tamano'], as_index=False, observed=True)
        .agg({'total_hormigas': 'mean'})
    )

    chart_temp = (
        alt.Chart(agrupado_temp)
        .mark_line(point=True)
        .encode(
            x=alt.X('temp_mid:Q', title='Temperatura media (°C)'),
            y=alt.Y('total_hormigas:Q', title='Actividad Promedio (Hormigas)'),
            color=alt.Color('categoria_tamano:N', title='Tamaño de Hormiga',
                            scale=alt.Scale(scheme='category10'),
                            legend=alt.Legend(orient='top')),
            tooltip=[
                alt.Tooltip('temp_mid:Q', title='Temp Aprox', format='.1f'),
                alt.Tooltip('categoria_tamano:N', title='Tamaño'),
                alt.Tooltip('total_hormigas:Q', title='Actividad Promedio', format=',.1f')
            ]
        )
        .properties(
            title={"text": "Evolución de Actividad Promedio por Temperatura",
                   "subtitle": "Comparación entre tamaños de hormigas (media de actividad)"}
        )
        .interactive()
    )
    return chart_temp


def get_altair_boxplot_velocidad(df):
    vel_cols = [c for c in df.columns if
                c in ['vel_en_cargadas', 'vel_en_sin_carga', 'vel_sn_cargadas', 'vel_sn_sin_carga']]
    if not vel_cols:
        return None

    df_melt = (
        df.melt(value_vars=vel_cols, var_name='tipo', value_name='velocidad')
        .dropna(subset=['velocidad'])
    )

    df_melt['carga'] = df_melt['tipo'].apply(lambda x: 'Con carga' if 'cargadas' in x else 'Sin carga')

    chart_vel = (
        alt.Chart(df_melt)
        .mark_boxplot(size=80, median={'color': 'black'})
        .encode(
            y=alt.Y('carga:N', title='Tipo de carga', sort=['Con carga', 'Sin carga']),
            x=alt.X('velocidad:Q', title='Velocidad promedio (mm/s)'),
            color=alt.Color('carga:N', scale=alt.Scale(domain=['Con carga', 'Sin carga'], range=['#d73027', '#fee08b']),
                            legend=None),
            tooltip=[alt.Tooltip('carga:N', title='Tipo de carga'),
                     alt.Tooltip('velocidad:Q', title='Velocidad promedio', format='.2f')]
        )
        .properties(
            title={"text": "Comparación de Velocidad de Hormigas según Carga",
                   "subtitle": "Distribución de velocidades combinando entrada y salida del nido"},
            height=400
        )
    )
    return chart_vel


def get_altair_scatter_temp_tamano(df, temp_range=None, size_range=None):
    columnas_necesarias = ['temp_media', 'tamano_promedio', 'categoria_tamano']
    columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]
    if columnas_faltantes:
        st.error(f"Error en Gráfico Tamaño: Faltan columnas: {columnas_faltantes}")
        return None

    df_plot = df.dropna(subset=['temp_media', 'tamano_promedio', 'categoria_tamano'])
    if df_plot.empty:
        st.warning(f"No hay datos válidos (con Temp, Tamaño y Categoría) para mostrar en la selección actual.")
        return None

    if temp_range:
        x_axis_encoding = alt.X('temp_media:Q', title='Temperatura media (°C)', scale=alt.Scale(domain=temp_range))
    else:
        x_axis_encoding = alt.X('temp_media:Q', title='Temperatura media (°C)')

    if size_range:
        y_axis_encoding = alt.Y('tamano_promedio:Q', title='Tamaño corporal promedio (área mm²)',
                                scale=alt.Scale(domain=size_range))
    else:
        y_axis_encoding = alt.Y('tamano_promedio:Q', title='Tamaño corporal promedio (área mm²)')

    chart_scatter = (
        alt.Chart(df_plot)
        .mark_circle(opacity=0.4, size=60)
        .encode(
            x=x_axis_encoding,
            y=y_axis_encoding,
            color=alt.Color('categoria_tamano:N', title='Grupo de tamaño',
                            scale=alt.Scale(domain=['Pequeñas', 'Medianas', 'Grandes'],
                                            range=['#fee08b', '#f46d43', '#d73027'])),
            tooltip=[
                alt.Tooltip('temp_media:Q', title='Temperatura (°C)', format='.1f'),
                alt.Tooltip('tamano_promedio:Q', title='Tamaño promedio', format='.2f'),
                alt.Tooltip('categoria_tamano:N', title='Grupo')
            ]
        )
        .properties(title={"text": "Relación entre Temperatura y Tamaño Corporal"})
        .interactive()
    )
    return chart_scatter


# --- Funciones de Machine Learning ---

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors="ignore")


@st.cache_resource
def train_models(df):
    # --- RE-PROCESAMIENTO PARA ML MEJORADO ---
    # 1. Definir la nueva variable objetivo según los nuevos rangos
    # Nula: 0 | Baja: 1-10 | Media: 11-40 | Alta: >40

    # Recalcular total cargadas para asegurar consistencia
    cols_en_cargadas = [c for c in ['en_cargadas'] if c in df.columns]
    cols_sn_cargadas = [c for c in ['sn_cargadas'] if c in df.columns]

    # Trabajamos con una copia para no afectar el dataframe global en cache si se vuelve a usar
    X = df.copy()
    X['total_hormigas_cargadas'] = X[cols_en_cargadas].sum(axis=1, skipna=True) + X[cols_sn_cargadas].sum(axis=1,
                                                                                                          skipna=True)

    bins = [-float('inf'), 0, 10, 40, float('inf')]
    labels = ["Nula", "Baja", "Media", "Alta"]

    # Actualizamos las etiquetas globales para la UI
    st.session_state.class_labels = labels

    X['Actividad_Clase'] = pd.cut(
        X['total_hormigas_cargadas'].fillna(0),
        bins=bins,
        labels=labels,
        right=True
    )

    # Definir target (shift -1 para predecir futuro inmediato)
    y = X['Actividad_Clase'].shift(-1)

    # Eliminar filas inválidas por el shift y resetear
    X = X.iloc[2:-1].reset_index(drop=True)
    y = y.iloc[2:-1].reset_index(drop=True)

    # Eliminar nulos en target
    valid_indices = y.dropna().index
    X = X.loc[valid_indices].reset_index(drop=True)
    y = y.loc[valid_indices].reset_index(drop=True)

    y = y.astype(pd.CategoricalDtype(categories=labels))

    # --- SPLIT TIME SERIES ---
    # Usamos TimeSeriesSplit como se solicitó
    tscv = TimeSeriesSplit(n_splits=5)

    # Obtenemos el último split para entrenar/testear de forma secuencial
    # Esto simula el comportamiento de entrenar con el pasado y evaluar en el futuro más reciente
    train_index, test_index = list(tscv.split(X))[-1]

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    st.session_state.ml_info = {
        "dia_test": "Último bloque temporal (TimeSeriesSplit)",
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    # --- PIPELINE ---
    cols_to_drop = [col for col in X.columns if col not in FEATURES_NUMERICAS_ML]
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])

    preprocessor = Pipeline([
        ("drop_cols", DropColumns(columns_to_drop=cols_to_drop)),
        ("column_transformer", ColumnTransformer(
            [("num", numeric_pipeline, FEATURES_NUMERICAS_ML)],
            remainder="drop"
        ))
    ])

    # Modelo SVM con los parámetros fijos solicitados: C=270, gamma='scale', kernel='rbf'
    modelo_svm = SVC(
        C=270,
        gamma="scale",
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42
    )

    pipeline_final = Pipeline([
        ("preprocessing", preprocessor),
        ("model", modelo_svm)
    ])

    # Entrenar
    pipeline_final.fit(X_train, y_train)
    preds = pipeline_final.predict(X_test)

    # Métricas
    report_dict = classification_report(
        y_test, preds, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, preds, labels=labels)
    f1_w = f1_score(y_test, preds, average='weighted')

    resultados_metricas = {
        "report_df": pd.DataFrame(report_dict).transpose(),
        "cm": cm,
        "f1_weighted": f1_w
    }

    return pipeline_final, resultados_metricas, X_train, y_train


# --- Carga de Datos ---
df_processed = load_and_process_data(DATA_FOLDER)

# --- Barra Lateral ---
st.sidebar.title("🐜 App de Análisis de Hormigas")
st.sidebar.markdown("---")

if df_processed.empty:
    st.sidebar.error("No se pudieron cargar datos. La aplicación está en modo limitado.")
    pagina = st.sidebar.radio("Navegación", ["Inicio"])
else:
    pagina = st.sidebar.radio("Navegación", ["Inicio", "Exploración de Datos (EDA)", "Modelo Predictivo (ML)"])

st.sidebar.markdown("---")
st.sidebar.markdown("Proyecto final de análisis de datos.")

# --- Inicio ---
if pagina == "Inicio":
    st.title("🐜 Análisis de Actividad y Modelo Predictivo")
    st.markdown("""
    Bienvenido a la aplicación de análisis de actividad de hormigas.
    ### Objetivos
    1.  **Explorar Datos:** Visualizar la dinámica histórica de entrada/salida y temperatura.
    2.  **Probar el Modelo:** Interactuar con un modelo de ML para predecir actividad "Alta" o "Baja".
    3.  **Validar Nuevos Datos:** Cargar y validar nuevos datasets.
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
        st.error("No se han podido cargar los datos de la carpeta `datos/`.")


# --- EDA ---
elif pagina == "Exploración de Datos (EDA)":
    st.title("📈 Exploración de Datos (EDA)")
    st.markdown("Visualización de los patrones de actividad de las hormigas.")

    if df_processed.empty:
        st.error("No se pueden mostrar gráficos porque no se cargaron datos.")
    else:
        st.subheader("Análisis de Distribuciones (Altair)")

        # Configuración de Filtros para Gráfico 1
        col_filters_1, col_filters_2 = st.columns(2)

        df_to_plot_chart1 = df_processed.copy()
        selected_file_label = "Todos los archivos"

        # Filtro 1: Selección de Archivo
        with col_filters_1:
            if 'source_file' in df_processed.columns:
                file_list = sorted(df_processed['source_file'].unique())
                options = ["Todos los archivos"] + file_list
                selected_file = st.selectbox("Filtrar por archivo (Chart 1):", options=options, index=0)
                if selected_file != "Todos los archivos":
                    df_to_plot_chart1 = df_processed[df_processed['source_file'] == selected_file]
                    selected_file_label = selected_file
            else:
                st.warning("No se encontró columna 'source_file' para filtro.")

        # Filtro 2: Variable Eje X (Nuevo)
        with col_filters_2:
            x_axis_options = {
                "Temperatura Media (°C)": "temp_media",
                "Humedad Relativa (%)": "hum_rel_media",
                "Radiación Solar (W/m²)": "rad_solar_media"
            }
            selected_x_label = st.selectbox("Variable Eje X:", list(x_axis_options.keys()))
            selected_x_col = x_axis_options[selected_x_label]

        # Calcular rangos dinámicos basados en la selección
        if not df_to_plot_chart1.empty and selected_x_col in df_to_plot_chart1.columns:
            x_min = float(df_processed[selected_x_col].min())  # Rango global para mantener consistencia
            x_max = float(df_processed[selected_x_col].max())
            x_domain = [x_min, x_max]
        else:
            x_domain = None

        global_ants_min = float(df_processed['total_hormigas'].min())
        global_ants_max = float(df_processed['total_hormigas'].max())
        ants_domain = [global_ants_min, global_ants_max]

        # Generar Gráfico 1 Modificado
        with st.spinner(f"Generando Scatter: {selected_x_label} vs. Actividad ({selected_file_label})..."):
            chart1 = get_altair_scatter_variable_actividad(
                df_to_plot_chart1,
                x_col=selected_x_col,
                x_title=selected_x_label,
                x_range=x_domain,
                ants_range=ants_domain
            )
            if chart1: st.altair_chart(chart1, use_container_width=True)

        st.markdown("---")

        # Gráfico 2 (Modificado: Line Chart)
        with st.spinner("Generando Gráfico de Línea: Actividad Media vs Temperatura..."):
            chart2 = get_altair_line_temp_tamano(df_processed)
            if chart2:
                st.altair_chart(chart2, use_container_width=True)
            else:
                st.warning("No se pudo generar el Gráfico Temp/Tamaño.")

        st.markdown("---")

        # Gráfico 3
        g3_temp_domain = [df_processed['temp_media'].min(), df_processed['temp_media'].max()]
        g3_size_domain = [df_processed['tamano_promedio'].min(), df_processed['tamano_promedio'].max()]

        df_to_plot_chart3 = df_processed.copy()
        selected_file_label_g3 = "Todos los archivos"

        if 'source_file' in df_processed.columns:
            options_g3 = ["Todos los archivos"] + sorted(df_processed['source_file'].unique())
            selected_file_g3 = st.selectbox("Filtrar por archivo (Temp vs. Tamaño):", options=options_g3, index=0,
                                            key='selectbox_chart_3')
            if selected_file_g3 != "Todos los archivos":
                df_to_plot_chart3 = df_processed[df_processed['source_file'] == selected_file_g3]
                selected_file_label_g3 = selected_file_g3

        with st.spinner(f"Generando Scatter: Temperatura vs. Tamaño ({selected_file_label_g3})..."):
            chart3 = get_altair_scatter_temp_tamano(df_to_plot_chart3, temp_range=g3_temp_domain,
                                                    size_range=g3_size_domain)
            if chart3: st.altair_chart(chart3, use_container_width=True)

        st.markdown("---")

        # Gráfico 4 (Modificado: Line Chart)
        with st.spinner("Generando Gráfico de Línea: Actividad Media vs Hora..."):
            chart4 = get_altair_line_hora_tamano(df_processed)
            if chart4:
                st.altair_chart(chart4, use_container_width=True)
            else:
                st.warning("No se pudo generar el Gráfico Hora/Tamaño.")

        st.markdown("---")

        # Gráfico 5
        with st.spinner("Generando Boxplot: Velocidad por Tipo de Carga..."):
            chart5 = get_altair_boxplot_velocidad(df_processed)
            if chart5:
                st.altair_chart(chart5, use_container_width=True)
            else:
                st.warning("No se pudo generar el Boxplot Velocidad.")


# --- Modelo Predictivo (ML) ---
elif pagina == "Modelo Predictivo (ML)":

    st.title("🤖 Modelo Predictivo (ML)")

    if df_processed.empty:
        st.error("No se pueden entrenar modelos porque no se cargaron datos.")
    else:
        with st.spinner("Entrenando modelo SVM... (Esto puede tardar un momento la primera vez)"):
            pipeline_prediccion, metricas, X_train_data, y_train_data = train_models(df_processed)

        if pipeline_prediccion is None:
            st.error("Falló el entrenamiento del modelo.")
        else:
            model_classes = pipeline_prediccion.classes_
            st.session_state.model_classes = model_classes

            st.subheader("Prueba de Predicción y Explicación (LIME)")

            # --- ACTUALIZACIÓN DE LA LEYENDA ---
            st.info("""
            **Rangos de Actividad (hormigas cargadas por minuto):**
            - **Nula:** 0 
            - **Baja:** 1 - 10 
            - **Media:** 11 - 40 
            - **Alta:** > 40
            """)

            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
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
                    # --- ELIMINADO: input_precip ---

                submitted = st.form_submit_button("Predecir y Explicar")

            if submitted:
                # 1. Crear DataFrame para predicción sin precipitación
                input_data = pd.DataFrame({
                    'temp_media': [input_temp],
                    'rad_solar_media': [input_rad],
                    'hum_rel_media': [input_hum]
                })

                # Rellenamos columnas faltantes para que el Pipeline no se queje (aunque las ignore)
                for col in X_train_data.columns:
                    if col not in input_data.columns:
                        input_data[col] = 0

                try:
                    model_classes = st.session_state.model_classes

                    # --- CORRECCIÓN CRÍTICA: Usar predict() en vez de argmax(predict_proba) ---
                    # Las SVM con class_weight='balanced' y probability=True suelen dar probabilidades descalibradas
                    # (Platt scaling suave) que contradicen la decisión geométrica del hiperplano.
                    # Usamos predict() para respetar el balanceo y que "salten" más las clases minoritarias (Alta).

                    prediccion = pipeline_prediccion.predict(input_data)[0]

                    # Recuperamos la probabilidad asociada a esa clase predicha solo para mostrarla
                    # (aunque sea baja, es la decisión correcta del SVM balanceado)
                    class_index = np.where(model_classes == prediccion)[0][0]
                    probabilidades = pipeline_prediccion.predict_proba(input_data)[0]
                    confianza_predicha = probabilidades[class_index]

                    # Índice numérico para LIME y gráficos
                    predicted_class_index = class_index

                    st.markdown("---")
                    st.subheader("Resultado de la Predicción:")

                    if prediccion == "Alta":
                        emoji = "🐜"
                    elif prediccion == "Media":
                        emoji = "🚶"
                    elif prediccion == "Baja":
                        emoji = "📉"
                    else:
                        emoji = "🛑"

                    st.metric("Nivel de Actividad", f"{emoji} {prediccion}",
                              f"{confianza_predicha * 100:.1f}% de confianza")

                    # --- NUEVO: Gráfico de Probabilidades ---
                    st.markdown("##### Probabilidades por clase")
                    probs_df = pd.DataFrame({
                        'Clase': model_classes,
                        'Probabilidad': probabilidades
                    })

                    # Asegurar orden lógico en el gráfico
                    orden_clases = ["Nula", "Baja", "Media", "Alta"]

                    chart_probs = (
                        alt.Chart(probs_df)
                        .mark_bar()
                        .encode(
                            x=alt.X('Clase', sort=orden_clases, title=None),
                            y=alt.Y('Probabilidad', title='Probabilidad', scale=alt.Scale(domain=[0, 1])),
                            color=alt.Color('Clase', legend=None, scale=alt.Scale(domain=orden_clases,
                                                                                  range=['#d3d3d3', '#fee08b',
                                                                                         '#fc8d59', '#d73027'])),
                            tooltip=[alt.Tooltip('Clase'), alt.Tooltip('Probabilidad', format='.1%')]
                        )
                        .properties(height=200)
                    )
                    st.altair_chart(chart_probs, use_container_width=True)

                    # --- CAMBIO: LIME EN LUGAR DE SHAP ---
                    st.subheader("Explicación Local (LIME)")
                    st.markdown(f"¿Por qué el modelo eligió **{prediccion}**?")

                    # Crear explicador LIME (Usando solo columnas numéricas y asegurando float)
                    # Convertimos a float para evitar problemas de tipos mixtos
                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        training_data=X_train_data[FEATURES_NUMERICAS_ML].astype(float).values,
                        feature_names=FEATURES_NUMERICAS_ML,
                        class_names=list(model_classes),
                        mode='classification',
                        verbose=False
                    )


                    # Función wrapper para LIME
                    def predict_fn_lime(numpy_array):
                        df_temp = pd.DataFrame(numpy_array, columns=FEATURES_NUMERICAS_ML)
                        return pipeline_prediccion.predict_proba(df_temp)


                    data_row_lime = input_data[FEATURES_NUMERICAS_ML].astype(float).values[0]

                    exp = explainer.explain_instance(
                        data_row=data_row_lime,
                        predict_fn=predict_fn_lime,
                        num_features=len(FEATURES_NUMERICAS_ML),  # Usar todas las features disponibles (ahora son 3)
                        labels=[predicted_class_index]  # Explicar explícitamente la clase ganadora
                    )

                    # Mostrar gráfico forzando la etiqueta correcta y ajustando tamaño
                    c1, c2 = st.columns([3, 2])  # La columna izquierda (c1) ocupará el 60% del ancho

                    with c1:  # Renderizamos en la columna restringida
                        with plt.style.context("ggplot"):
                            # 1. Generar figura
                            fig_lime = exp.as_pyplot_figure(label=predicted_class_index)

                            # 2. Tamaño: Dale ANCHO para que entre el texto, pero poca ALTURA
                            # (6, 3) suele ser un buen balance. (5, 2.5) es más compacto.
                            fig_lime.set_size_inches(6, 3)

                            ax = fig_lime.get_axes()[0]

                            # 3. Fuentes legibles (ya no uses 3 o 4)
                            ax.set_title(f"Factores para predicción: {prediccion}", fontsize=10)
                            ax.tick_params(axis='y', labelsize=8)  # Texto de las variables
                            ax.tick_params(axis='x', labelsize=7)  # Números de abajo
                            ax.set_xlabel("Contribución a la probabilidad", fontsize=8)

                            # Quitar bordes innecesarios para limpiar la visual
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)

                            # 4. Ajuste importante para que no se corte el texto de la izquierda
                            plt.tight_layout()

                            # 5. Renderizar en Streamlit
                            # use_container_width=True ajustará la imagen al ancho de 'c1'
                            st.pyplot(fig_lime, use_container_width=True, bbox_inches='tight')

                    with st.expander("Ver detalles numéricos de la explicación"):
                        st.write(exp.as_list(label=predicted_class_index))

                except Exception as e:
                    st.error(f"Error durante la predicción: {e}")
                    # Mostrar input_data para debug (opcional)
                    # st.dataframe(input_data)