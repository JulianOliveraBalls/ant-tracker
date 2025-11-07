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
        st.error(f"Error: El directorio '{folder_path}' no se encontró. Asegúrate de que exista en el repositorio de GitHub.")
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

        df_processed['total_entrando'] = df_processed[cols_en_cargadas].sum(axis=1) + df_processed[cols_en_sin_carga].sum(axis=1)
        df_processed['total_saliendo'] = df_processed[cols_sn_cargadas].sum(axis=1) + df_processed[cols_sn_sin_carga].sum(axis=1)
        df_processed['total_cargadas'] = df_processed[cols_en_cargadas].sum(axis=1) + df_processed[cols_sn_cargadas].sum(axis=1)
        df_processed['total_sin_carga'] = df_processed[cols_en_sin_carga].sum(axis=1) + df_processed[cols_sn_sin_carga].sum(axis=1)
        
        df_processed['total_hormigas'] = df_processed['total_entrando'] + df_processed['total_saliendo']
        
        # Feature Engineering para ML
        cuartil_superior = df_processed['total_cargadas'].quantile(0.75)
        st.session_state.cuartil_superior = cuartil_superior # Guardar para referencia
        df_processed['AltaActividad'] = (df_processed['total_cargadas'] > cuartil_superior).astype(int)

        # Rellenar nulos en columnas de movimiento
        cols_mov_existentes = [c for c in COLS_MOVIMIENTO if c in df_processed.columns]
        df_processed[cols_mov_existentes] = df_processed[cols_mov_existentes].fillna(0)
        
        # Feature Engineering para gráficos Altair
        cols_area = [c for c in ['area_en_cargadas', 'area_en_sin_carga', 'area_sn_cargadas', 'area_sn_sin_carga'] if c in df_processed.columns]
        df_processed['tamano_promedio'] = df_processed[cols_area].mean(axis=1, skipna=True)
        
        if not df_processed.dropna(subset=['tamano_promedio']).empty:
             df_processed['categoria_tamano'] = pd.qcut(df_processed.dropna(subset=['tamano_promedio'])['tamano_promedio'], 3, labels=['Pequeñas', 'Medianas', 'Grandes'])

        return df_processed

    except Exception as e:
        st.error(f"Error durante el feature engineering: {e}")
        st.dataframe(df_processed.head()) # Muestra dónde falló
        return pd.DataFrame()


# --- Funciones de Gráficos (Adaptadas para usar nombres de columna nuevos) ---

# (Todas las funciones de Matplotlib/Seaborn han sido eliminadas:
# plot_totales_simple, plot_cargadas_vs_sin, plot_entrando_saliendo_cargadas,
# plot_correlacion_temperatura, plot_area_vs_velocidad, plot_heatmap_tamaño_hora)


# --- Funciones de Gráficos (Altair) ---

def get_altair_chart_temp_actividad(df):
    """Altair Scatter: Temperatura vs Actividad Total."""
    if 'temp_media' not in df.columns or 'total_hormigas' not in df.columns or 'fecha_hora_sensor' not in df.columns:
        return None

    df_plot = df[(df['temp_media'] >= 0) & (df['temp_media'] <= 40) & (df['total_hormigas'] >= 0)]

    chart = (
        alt.Chart(df_plot)
        .mark_circle(size=80, opacity=0.55, color='#2563eb', stroke='#1e3a8a', strokeWidth=0.6)
        .encode(
            x=alt.X('temp_media:Q', title='Temperatura media (°C)'),
            y=alt.Y('total_hormigas:Q', title='Total de hormigas'),
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

def get_altair_chart_hora_actividad(df):
    """Altair Scatter: Hora vs Actividad Total."""
    if 'hora' not in df.columns or 'total_hormigas' not in df.columns or 'fecha_hora_sensor' not in df.columns:
        return None

    df_plot = df[(df['hora'] >= 0) & (df['hora'] <= 23) & (df['total_hormigas'] >= 0)]
    
    chart = (
        alt.Chart(df_plot)
        .mark_circle(size=80, opacity=0.55, color='#10b981', stroke='#065f46', strokeWidth=0.6)
        .encode(
            x=alt.X('hora:Q', title='Hora del día', scale=alt.Scale(domain=[0, 23])),
            y=alt.Y('total_hormigas:Q', title='Total de hormigas (entrada + salida)'),
            tooltip=[
                alt.Tooltip('fecha_hora_sensor:T', title='Fecha y hora'),
                alt.Tooltip('hora:Q', title='Hora del día'),
                alt.Tooltip('total_hormigas:Q', title='Hormigas totales', format=",.0f")
            ]
        )
        .properties(
            title={
                "text": "Actividad de Hormigas según Hora del Día",
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

    agrupado = (
        df.groupby(['hora', 'categoria_tamano'], as_index=False, observed=True)
        .agg({'total_hormigas': 'sum'})
    )

    chart = (
        alt.Chart(agrupado)
        .mark_rect(strokeWidth=0)
        .encode(
            x=alt.X('hora:O', title='Hora del día', sort=list(range(24)), axis=alt.Axis(labelAngle=0)),
            y=alt.Y('categoria_tamano:N', title='Tamaño de hormigas', sort=['Pequeñas', 'Medianas', 'Grandes']),
            color=alt.Color(
                'total_hormigas:Q',
                title='Cantidad total de hormigas',
                scale=alt.Scale(scheme='goldred')
            ),
            tooltip=[
                alt.Tooltip('hora:O', title='Hora'),
                alt.Tooltip('categoria_tamano:N', title='Tamaño'),
                alt.Tooltip('total_hormigas:Q', title='Total de hormigas', format=',.0f')
            ]
        )
        .properties(
            title={
                "text": "Mapa de Calor de Actividad Total de Hormigas por Hora",
                "subtitle": "Suma de entradas y salidas agrupado por hora y tamaño corporal"
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

    agrupado_temp = (
        df_plot.groupby(['temp_bin', 'categoria_tamano'], as_index=False, observed=True)
        .agg({'total_hormigas': 'sum'})
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
                title='Cantidad total de hormigas',
                scale=alt.Scale(scheme='goldred')
            ),
            tooltip=[
                alt.Tooltip('temp_label:O', title='Temperatura (°C)'),
                alt.Tooltip('categoria_tamano:N', title='Tamaño'),
                alt.Tooltip('total_hormigas:Q', title='Total de hormigas', format=',.0f')
            ]
        )
        .properties(
            title={
                "text": "Mapa de Calor de Actividad Total de Hormigas según Temperatura",
                "subtitle": "Suma de entradas y salidas agrupado por temperatura y tamaño corporal"
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
            }
        )
    )
    return chart_vel

def get_altair_scatter_temp_tamano(df):
    """Altair Scatter: Temperatura vs Tamaño."""
    if 'temp_media' not in df.columns or 'tamano_promedio' not in df.columns or 'categoria_tamano' not in df.columns:
        return None
        
    chart_scatter = (
        alt.Chart(df.dropna(subset=['temp_media', 'tamano_promedio', 'categoria_tamano']))
        .mark_circle(opacity=0.4, size=60)
        .encode(
            x=alt.X('temp_media:Q', title='Temperatura media (°C)'),
            y=alt.Y('tamano_promedio:Q', title='Tamaño corporal promedio (área mm²)'),
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
    Entrena los modelos de ML y devuelve los pipelines y resultados.
    """
    
    # 1. Preparar datos para ML
    y = df['AltaActividad'].shift(-1)
    X = df.copy()
    
    # Recortar X e y para alinear (basado en script ML)
    X = X.iloc[2:-1].reset_index(drop=True)
    y = y.iloc[2:-1].reset_index(drop=True)
    
    # 2. Separación Train/Test temporal
    # (Asumiendo que los archivos tienen fechas diferentes.
    # Si todos son del mismo día, esto fallará.)
    dias_unicos = X['dia_str'].unique()
    dias_unicos.sort()
    
    if len(dias_unicos) < 2:
        st.error("Error de ML: Se necesita data de al menos 2 días diferentes para hacer la partición Train/Test temporal.")
        return None, None
    
    DIA_DE_TEST = dias_unicos[-1] # Usar el último día para test
    train_mask = X['dia_str'] != DIA_DE_TEST
    test_mask = X['dia_str'] == DIA_DE_TEST

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    st.session_state.ml_info = {
        "dia_test": DIA_DE_TEST,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    # 3. Pipeline de Preprocesamiento
    
    # Columnas a eliminar (todas las que no son features)
    cols_to_drop = [col for col in X.columns if col not in FEATURES_NUMERICAS_ML]
    
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])

    preprocessor = Pipeline([
        ("drop_cols", DropColumns(columns_to_drop=cols_to_drop)),
        ("column_transformer", ColumnTransformer(
            [("num", numeric_pipeline, FEATURES_NUMERICAS_ML)],
            remainder="drop"
        ))
    ])

    # 4. Definición y Entrenamiento de Modelos
    modelos_definicion = {
        "SVM": SVC(C=0.3058, gamma="auto", kernel="rbf", class_weight="balanced", probability=True),
        "LogisticRegression": LogisticRegression(
            C=3.7554, solver="lbfgs", max_iter=1000, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=376, max_depth=6, min_samples_leaf=13,
            max_features="log2", random_state=42, class_weight="balanced"
        ),
    }
    
    pipelines_entrenados = {}
    resultados_metricas = {}

    for nombre, modelo in modelos_definicion.items():
        pipeline_final = Pipeline([
            ("preprocessing", preprocessor),
            ("model", modelo)
        ])
        
        pipeline_final.fit(X_train, y_train)
        preds = pipeline_final.predict(X_test)
        
        # FIX: Añadido labels=[0, 1] para asegurar que ambas clases estén en el reporte
        report_dict = classification_report(
            y_test, preds, labels=[0, 1], output_dict=True, zero_division=0
        )
        # FIX: Añadir labels=[0, 1] aquí también para forzar la matriz 2x2
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        
        resultados_metricas[nombre] = {
            "report_df": pd.DataFrame(report_dict).transpose(),
            "cm": cm,
            "f1": f1_score(y_test, preds)
        }
        pipelines_entrenados[nombre] = pipeline_final
        
    return pipelines_entrenados, resultados_metricas

def get_feature_importance(pipeline, X_test, y_test):
    """Calcula y devuelve la importancia de las features."""
    
    feature_names = FEATURES_NUMERICAS_ML
    modelo = pipeline.named_steps['model']
    
    importances_data = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        
        # Random Forest
        if hasattr(modelo, "feature_importances_"):
            importances = modelo.feature_importances_
            importances_data = {f: imp for f, imp in zip(feature_names, importances)}
            
        # Logistic Regression
        elif hasattr(modelo, "coef_"):
            coefs = modelo.coef_[0]
            importances_data = {f: c for f, c in zip(feature_names, coefs)}
            
        # SVM (Permutation Importance)
        elif isinstance(modelo, SVC) and modelo.kernel == 'rbf':
            preproc = pipeline.named_steps['preprocessing']
            X_test_transformed = preproc.transform(X_test)
                
            perm_result = permutation_importance(
                modelo, X_test_transformed, y_test,
                n_repeats=5, random_state=42, n_jobs=-1, scoring='f1'
            )
            importances = perm_result.importances_mean
            importances_data = {f: imp for f, imp in zip(feature_names, importances)}
            
    if not importances_data:
        return pd.DataFrame(columns=["Feature", "Importancia"])

    df_imp = pd.DataFrame(
        list(importances_data.items()), 
        columns=["Feature", "Importancia"]
    )
    
    # Añadir valor absoluto para ordenar
    df_imp["Importancia_Abs"] = df_imp["Importancia"].abs()
    df_imp = df_imp.sort_values(by="Importancia_Abs", ascending=False)
    
    return df_imp


# --- Funciones de Validación de Archivos ---

def validar_archivo_subido(df):
    """
    Valida un DataFrame subido contra las columnas requeridas y busca nulos.
    """
    errores = []

    # 1. Validar columnas
    # FIX: Usar la lista completa de 34 columnas que proporcionaste
    COLUMNAS_VALIDACION_COMPLETAS = set([
        'hora_inicio', 'hora_fin', 'total_hormigas_entrando_al_nido_en_cargadas',
        'total_hormigas_entrando_al_nido_en_sin_carga', 'total_hormigas_saliendo_del_nido_sn_cargadas',
        'total_hormigas_saliendo_del_nido_sn_sin_carga', 'velocidad_promedio_en__cargadas',
        'velocidad_promedio_en__sin_carga', 'velocidad_promedio_sn__cargadas',
        'velocidad_promedio_sn__sin_carga', 'rea_mediana_en__cargadas',
        'rea_mediana_en__sin_carga', 'rea_mediana_sn__cargadas', 'rea_mediana_sn__sin_carga',
        'largo_mediana_en__cargadas', 'largo_mediana_en__sin_carga',
        'largo_mediana_sn__cargadas', 'largo_mediana_sn__sin_carga',
        'ancho_mediana_en__cargadas', 'ancho_mediana_en__sin_carga',
        'ancho_mediana_sn__cargadas', 'ancho_mediana_sn__sin_carga',
        'fecha_hora_sensor', 'temp_media', 'temp_max', 'temp_mín',
        'dew point_media', 'dew point_mín', 'solar radiation dgt_media',
        'relative humidity_media', 'relative humidity_max', 'relative humidity_mín',
        'precip_total', 'battery voltage_última'
    ])
    
    columnas_en_df = set(df.columns)
    
    columnas_faltantes = COLUMNAS_VALIDACION_COMPLETAS - columnas_en_df
    columnas_encontradas = COLUMNAS_VALIDACION_COMPLETAS.intersection(columnas_en_df)
    
    if columnas_faltantes:
        for col in columnas_faltantes:
            errores.append(f"Falta la columna requerida: {col}")

    # 2. Validar nulos (solo en columnas que sí existen y se encontraron)
    if columnas_encontradas:
        nulos = df[list(columnas_encontradas)].isnull().sum()
        columnas_con_nulos = nulos[nulos > 0]
        
        if not columnas_con_nulos.empty:
            for col, count in columnas_con_nulos.items():
                errores.append(f"La columna '{col}' tiene {count} valores nulos.")
            
    return errores


# --- Cuerpo Principal de la App Streamlit ---

# Cargar y procesar datos al inicio
df_processed = load_and_process_data(DATA_FOLDER)

# --- Barra Lateral (Navegación) ---
st.sidebar.title("🐜 App de Análisis de Hormigas")
st.sidebar.markdown("---")

if df_processed.empty:
    st.sidebar.error("No se pudieron cargar datos. La aplicación está en modo limitado.")
    pagina = st.sidebar.radio("Navegación", ["Inicio", "Cargar Nuevo Archivo"])
else:
     pagina = st.sidebar.radio("Navegación", ["Inicio", "Cargar Nuevo Archivo", "Exploración de Datos (EDA)", "Modelo Predictivo (ML)"])

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


# --- Página de Carga de Archivos ---
elif pagina == "Cargar Nuevo Archivo":
    st.title("📂 Cargar y Validar Nuevo Archivo de Datos")
    st.markdown("""
    Esta sección te permite subir un nuevo archivo de datos (CSV o XLSX) para validarlo 
    contra la estructura requerida por el modelo.
    
    **Importante:** Debido a las restricciones de seguridad de Streamlit Cloud, esta
    función **no guardará permanentemente** el archivo en el repositorio. Es solo
    un validador. Para añadir datos de forma permanente, sube el archivo a la
    carpeta `datos/` en tu repositorio de GitHub.
    """)

    # Inputs para el nombre del archivo
    st.subheader("1. Define la fecha del archivo")
    col1, col2, col3 = st.columns(3)
    with col1:
        ano = st.number_input("Año", min_value=2020, max_value=2030, value=datetime.date.today().year)
    with col2:
        mes = st.number_input("Mes", min_value=1, max_value=12, value=datetime.date.today().month)
    with col3:
        dia = st.number_input("Día", min_value=1, max_value=31, value=datetime.date.today().day)
        
    try:
        fecha_obj = datetime.date(ano, mes, dia)
        nombre_archivo_generado = f"{fecha_obj.strftime('%Y%m%d')}-tiempo_final.xlsx"
        st.info(f"Nombre de archivo generado: **{nombre_archivo_generado}**")
    except ValueError as e:
        st.error(f"Fecha inválida: {e}")
        nombre_archivo_generado = None

    # Comprobar duplicados
    if nombre_archivo_generado and os.path.exists(DATA_FOLDER):
        # FIX: Comprobación más robusta (ignora extensión)
        nombre_base_generado = os.path.splitext(nombre_archivo_generado)[0]
        archivos_existentes_base = [os.path.splitext(f)[0] for f in os.listdir(DATA_FOLDER) if f.endswith(('.xlsx', '.xls'))]
        
        if nombre_base_generado in archivos_existentes_base:
            st.error(f"**¡Atención!** Un archivo con la fecha base `{nombre_base_generado}` ya existe en la carpeta `datos/`. No se puede guardar un duplicado.")
            duplicado = True
        else:
            st.success("El nombre de archivo está disponible.")
            duplicado = False
    else:
        duplicado = False

    st.subheader("2. Sube tu archivo (.xlsx o .csv)")
    uploaded_file = st.file_uploader("Selecciona un archivo", type=['xlsx', 'csv'])

    st.subheader("3. Validar y (Simular) Guardar")
    if st.button("Validar Archivo", disabled=(uploaded_file is None or duplicado or nombre_archivo_generado is None)):
        with st.spinner("Validando archivo..."):
            df_subido = None
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_subido = pd.read_csv(uploaded_file)
                else:
                    df_subido = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")
            
            if df_subido is not None:
                errores = validar_archivo_subido(df_subido)
                
                if errores:
                    st.error("El archivo tiene errores y no puede ser procesado:")
                    for err in errores:
                        st.markdown(f"- {err}")
                else:
                    st.success(f"**¡Validación Exitosa!** El archivo `{uploaded_file.name}` tiene la estructura correcta.")
                    
                    # Preparar el archivo .xlsx en memoria
                    output = BytesIO()
                    if uploaded_file.name.endswith('.csv'):
                        st.info("El archivo .csv se convertirá a .xlsx.")
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_subido.to_excel(writer, index=False, sheet_name='Datos')
                    else:
                        # Si ya es xlsx, solo copia los bytes
                        uploaded_file.seek(0)
                        output.write(uploaded_file.read())

                    output.seek(0) # Reset buffer
                    
                    st.markdown(f"""
                    **Acción Requerida:** Para usar este archivo en la app, súbelo.
                    - **Opción 1 (Recomendada):** Descarga el archivo .xlsx y súbelo manualmente a la carpeta `datos/` de tu GitHub.
                    - **Opción 2 (Temporal):** Haz clic abajo para usar el archivo **solo en esta sesión**.
                    """)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label=f"Descargar {nombre_archivo_generado}",
                            data=output.getvalue(),
                            file_name=nombre_archivo_generado,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    with col2:
                        if st.button("Usar archivo en esta sesión (Temporal)"):
                            # Asegurarse de que la carpeta datos existe
                            os.makedirs(DATA_FOLDER, exist_ok=True)
                            
                            file_path = os.path.join(DATA_FOLDER, nombre_archivo_generado)
                            try:
                                with open(file_path, "wb") as f:
                                    f.write(output.getvalue())
                                
                                st.success(f"Archivo `{nombre_archivo_generado}` cargado temporalmente.")
                                st.warning("Este archivo se perderá si la app se reinicia. Súbelo a GitHub para guardarlo permanentemente.")
                                
                                # Limpiar la caché de datos y reiniciar
                                st.cache_data.clear()
                                st.rerun() # st.experimental_rerun() para versiones antiguas
                                
                            except Exception as e:
                                st.error(f"No se pudo guardar el archivo temporalmente: {e}")


# --- Página de Exploración de Datos (EDA) ---
elif pagina == "Exploración de Datos (EDA)":
    st.title("📈 Exploración de Datos (EDA)")
    st.markdown("Visualización de los patrones de actividad de las hormigas basados en todos los datos históricos cargados.")

    # Asegurarse de que los datos estén cargados
    if df_processed.empty:
        st.error("No se pueden mostrar gráficos porque no se cargaron datos.")
    else:
        # (Sección de Matplotlib/Seaborn eliminada)
        st.subheader("Análisis de Distribuciones (Altair)")
        st.markdown("Gráficos interactivos para explorar relaciones entre variables.")

        grafico_altair_tipo = st.selectbox(
            "Selecciona un gráfico (Altair):",
            [
                "Scatter: Temperatura vs. Actividad Total",
                "Scatter: Hora del Día vs. Actividad Total",
                "Boxplot: Velocidad por Tipo de Carga",
                "Scatter: Temperatura vs. Tamaño Corporal",
                "Heatmap: Actividad por Hora y Tamaño Corporal",
                "Heatmap: Actividad por Temperatura y Tamaño Corporal"
            ]
        )
        
        # Generar gráfico Altair seleccionado
        with st.spinner("Generando gráfico interactivo..."):
            chart_altair = None
            if grafico_altair_tipo == "Scatter: Temperatura vs. Actividad Total":
                chart_altair = get_altair_chart_temp_actividad(df_processed)
            elif grafico_altair_tipo == "Scatter: Hora del Día vs. Actividad Total":
                chart_altair = get_altair_chart_hora_actividad(df_processed)
            elif grafico_altair_tipo == "Boxplot: Velocidad por Tipo de Carga":
                chart_altair = get_altair_boxplot_velocidad(df_processed)
            elif grafico_altair_tipo == "Scatter: Temperatura vs. Tamaño Corporal":
                chart_altair = get_altair_scatter_temp_tamano(df_processed)
            elif grafico_altair_tipo == "Heatmap: Actividad por Hora y Tamaño Corporal":
                chart_altair = get_altair_heatmap_hora_tamano(df_processed)
            elif grafico_altair_tipo == "Heatmap: Actividad por Temperatura y Tamaño Corporal":
                chart_altair = get_altair_heatmap_temp_tamano(df_processed)

            # FIX 2: Centrar el gráfico usando columnas
            col_chart1, col_chart2, col_chart3 = st.columns([0.1, 0.8, 0.1]) # 10% 80% 10%
            
            with col_chart2:
                if chart_altair:
                    # Usar el ancho del contenedor DE LA COLUMNA CENTRAL
                    st.altair_chart(chart_altair, use_container_width=True)
                else:
                    st.warning("No se pudo generar el gráfico seleccionado. Verifica que las columnas necesarias (ej. 'categoria_tamano', 'tamano_promedio') existan.")


# --- Página de Modelo Predictivo (ML) ---
elif pagina == "Modelo Predictivo (ML)":
    st.title("🤖 Modelo Predictivo (ML)")
    st.markdown("""
    Esta sección detalla el rendimiento de los modelos de Machine Learning entrenados
    y ofrece una interfaz para probar el mejor modelo con datos nuevos.
    """)

    if df_processed.empty:
        st.error("No se pueden entrenar modelos porque no se cargaron datos.")
    else:
        # Entrenar modelos (usará caché si ya se ejecutó)
        with st.spinner("Entrenando modelos... (Esto puede tardar un momento la primera vez)"):
            pipelines, metricas = train_models(df_processed)

        if pipelines is None:
            st.error("Falló el entrenamiento del modelo. Revisa los mensajes de error anteriores.")
        else:
            ml_info = st.session_state.get('ml_info', {})
            
            # FIX: Formateo condicional para evitar ValueError
            umbral = st.session_state.get('cuartil_superior', 'N/A')
            umbral_str = f"{umbral:.1f}" if isinstance(umbral, (int, float)) else str(umbral)
            
            st.info(f"""
            **Información del Entrenamiento:**
            - **Datos de Entrenamiento:** {ml_info.get('train_samples', 'N/A')} registros.
            - **Datos de Prueba:** {ml_info.get('test_samples', 'N/A')} registros (Día: {ml_info.get('dia_test', 'N/A')}).
            - **Umbral de 'Alta Actividad':** > {umbral_str} hormigas cargadas/minuto.
            """)

            tab1, tab2, tab3 = st.tabs([
                "🧪 Probar el Modelo (Live)",
                "📊 Métricas de Rendimiento",
                "🌳 Importancia de Features"
            ])

            # --- Pestaña 1: Probar el Modelo ---
            with tab1:
                st.subheader("Prueba de Predicción en Tiempo Real")
                st.markdown("Ingresa las condiciones ambientales para predecir la actividad en el minuto siguiente.")
                
                # Seleccionar el mejor modelo (basado en F1)
                modelo_seleccionado_nombre = max(metricas, key=lambda k: metricas[k]['f1'])
                pipeline_prediccion = pipelines[modelo_seleccionado_nombre]
                
                st.success(f"Modelo seleccionado para predicción: **{modelo_seleccionado_nombre}** (Mejor F1-Score)")

                with st.form("prediction_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sliders basados en los rangos del DataFrame
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

                        # Precipitación suele ser 0, usar number input
                        input_precip = st.number_input("🌧️ Precipitación Total (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)

                    submitted = st.form_submit_button("Predecir Actividad")

                if submitted:
                    # Crear DataFrame para predicción
                    input_data = pd.DataFrame({
                        'temp_media': [input_temp],
                        'rad_solar_media': [input_rad],
                        'hum_rel_media': [input_hum],
                        'precip_total': [input_precip]
                    })
                    
                    # Añadir columnas dummy que el preprocesador espera (aunque las elimine)
                    for col in df_processed.columns:
                         if col not in input_data.columns:
                            input_data[col] = 0 # Valor placeholder
                    
                    # Predecir
                    try:
                        prediccion = pipeline_prediccion.predict(input_data)[0]
                        probabilidades = pipeline_prediccion.predict_proba(input_data)[0]
                        
                        prob_baja = probabilidades[0]
                        prob_alta = probabilidades[1]

                        st.markdown("---")
                        st.subheader("Resultado de la Predicción:")
                        
                        if prediccion == 1:
                            st.metric("Nivel de Actividad", "🐜 ALTA", f"{prob_alta*100:.1f}% de confianza")
                            st.warning("Se espera un alto flujo de hormigas cargadas.")
                        else:
                            st.metric("Nivel de Actividad", "📉 BAJA", f"{prob_baja*100:.1f}% de confianza")
                            st.info("Se espera un flujo normal o bajo de hormigas.")

                    except Exception as e:
                        st.error(f"Error durante la predicción: {e}")
                        st.dataframe(input_data) # Mostrar qué datos causaron el error

            # --- Pestaña 2: Métricas de Rendimiento ---
            with tab2:
                st.subheader("Rendimiento de Modelos (Datos de Test)")
                
                modelo_a_ver = st.selectbox("Selecciona un modelo para ver sus métricas:", metricas.keys())
                
                if modelo_a_ver:
                    metricas_modelo = metricas[modelo_a_ver]
                    
                    # FIX: Acceder al f1-score de forma segura (aunque 'labels' debería garantizarlo)
                    f1_clase_1 = 0.0
                    if '1' in metricas_modelo['report_df'].index:
                        f1_clase_1 = metricas_modelo['report_df'].loc['1', 'f1-score']
                    
                    st.metric(f"F1-Score (Clase 1: Alta Actividad)", f"{f1_clase_1:.3f}")
                    
                    st.markdown("#### Reporte de Clasificación")
                    # FIX 1: Añadir round(3) por si es un error de renderizado
                    st.dataframe(metricas_modelo['report_df'].round(3))
                    
                    st.markdown("#### Matriz de Confusión")
                    # FIX: Mostrar CM como tabla/dataframe en lugar de gráfico
                    st.text("0: Baja Actividad, 1: Alta Actividad")
                    cm_df = pd.DataFrame(
                        metricas_modelo['cm'],
                        index=[f"Verdadero {i}" for i in [0, 1]],
                        columns=[f"Predicción {i}" for i in [0, 1]]
                    )
                    st.dataframe(cm_df)

            # --- Pestaña 3: Importancia de Features ---
            with tab3:
                st.subheader("Importancia de las Características del Modelo")
                st.markdown("""
                ¿Qué variables ambientales son más importantes para las predicciones del modelo?
                - **RandomForest:** Muestra la "impureza" (Gini).
                - **LogisticRegression:** Muestra el "coeficiente" (magnitud del impacto).
                - **SVM (RBF):** Muestra la "Importancia por Permutación" (cuánto cae el F1-Score si se "rompe" la variable).
                
                **Nota:** Un impacto negativo (rojo) significa que la variable reduce la probabilidad de "Alta Actividad".
                """)
                
                modelo_imp = st.selectbox("Selecciona un modelo para ver la importancia de features:", pipelines.keys())
                
                if modelo_imp:
                    # Requerimos X_test, y_test (solo disponibles si se entrenó)
                    if 'ml_info' in st.session_state:
                        # Recargar datos de test (no se almacenan en caché de recursos)
                        df_ml = load_and_process_data(DATA_FOLDER)
                        y_ml = df_ml['AltaActividad'].shift(-1)
                        X_ml = df_ml.copy()
                        X_ml = X_ml.iloc[2:-1].reset_index(drop=True)
                        y_ml = y_ml.iloc[2:-1].reset_index(drop=True)
                        test_mask = X_ml['dia_str'] == st.session_state.ml_info['dia_test']
                        X_test_imp, y_test_imp = X_ml[test_mask], y_ml[test_mask]

                        df_importancia = get_feature_importance(pipelines[modelo_imp], X_test_imp, y_test_imp)
                        
                        # FIX 3: Mejorar el gráfico de importancia
                        
                        # Añadir color para positivo/negativo
                        df_importancia['Color'] = df_importancia['Importancia'].apply(lambda x: 'Positivo' if x >= 0 else 'Negativo')
                        
                        chart_imp = alt.Chart(df_importancia).mark_bar().encode(
                            # Ordenar por el valor absoluto
                            x=alt.X('Importancia:Q', title="Impacto en el Modelo"),
                            y=alt.Y('Feature:N', sort=alt.SortField("Importancia_Abs", op="max", order="descending")),
                            # Colorear barras basado en positivo/negativo
                            color=alt.Color('Color:N', 
                                            scale=alt.Scale(domain=['Positivo', 'Negativo'], 
                                                            range=['#10b981', '#f43f5e']),
                                            legend=alt.Legend(title="Impacto")
                                           ),
                            tooltip=['Feature', 'Importancia']
                        ).properties(
                            title=f"Importancia de Features para {modelo_imp}"
                        )
                        
                        # FIX 2: Centrar el gráfico
                        col_imp1, col_imp2, col_imp3 = st.columns([0.1, 0.8, 0.1])
                        with col_imp2:
                            st.altair_chart(chart_imp, use_container_width=True)
                    else:
                        st.warning("No se pueden calcular las importancias. Re-ejecutando...")