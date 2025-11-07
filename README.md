# 🐜 Aplicación Streamlit para Análisis de Actividad de Hormigas

Esta es una aplicación **Streamlit** diseñada para explorar datos de actividad de hormigas y probar un modelo predictivo de **Machine Learning**.

---

## 📁 Estructura del Repositorio

Para que esta aplicación funcione correctamente en **Streamlit Cloud**, tu repositorio de GitHub **debe tener la siguiente estructura:**

tu-repositorio/
│
├── datos/
│ ├── 2022-01-04.xlsx <-- Tu primer archivo de datos
│ ├── 2022-01-05.xlsx <-- Tu segundo archivo de datos
│ ├── 2022-01-06.xlsx <-- etc...
│
├── streamlit_app.py <-- Archivo principal de la app
├── requirements.txt <-- Dependencias del proyecto
└── README.md <-- Este archivo

markdown
Copiar código

---

## 🚀 Pasos para el Despliegue

### 1️⃣ Crear un Repositorio en GitHub
Si aún no lo has hecho, crea un **nuevo repositorio público** en GitHub.

### 2️⃣ Crear la Carpeta `datos/`
Dentro de tu repositorio, crea una carpeta llamada exactamente `datos`.

### 3️⃣ Subir tus Archivos de Datos
Sube **todos tus archivos .xlsx** (o `.xls`) originales a esta carpeta `datos/`.

### 4️⃣ Subir los Archivos de la App
Sube los archivos:
- `streamlit_app.py`
- `requirements.txt`

al **directorio raíz** de tu repositorio.

### 5️⃣ Desplegar en Streamlit Cloud
1. Ve a [Streamlit Cloud](https://streamlit.io/cloud)
2. Haz clic en **"New app"**
3. Conecta tu cuenta de GitHub y selecciona el repositorio que acabas de crear
4. Asegúrate de que el **Main file path** sea:
streamlit_app.py

yaml
Copiar código
5. Haz clic en **"Deploy!"**

---

## 📦 Cómo Funciona la Carga de Archivos

La aplicación leerá automáticamente **todos los archivos `.xlsx`** que encuentre en la carpeta `datos/` cada vez que se inicie.

La sección **"Cargar Nuevo Archivo"** en la aplicación es solo un **validador**, que te permite:
- Verificar si un archivo nuevo tiene el formato correcto.
- Comprobar las columnas y datos antes de incorporarlo al sistema.

---

## 🧩 Cómo Añadir un Nuevo Archivo Permanentemente

Para agregar un nuevo archivo de datos de forma permanente, seguí estos pasos:

1. Validá el archivo usando la app (opcional pero recomendado).  
2. Renombrá el archivo con el formato:  
AAAAMMDD-tiempo_final.xlsx

yaml
Copiar código
Ejemplo: `20241107-tiempo_final.xlsx`
3. Subí ese archivo a la carpeta `datos/` en tu repositorio de GitHub.  
4. Hacé un **commit** con los cambios.

> ⚡ Streamlit Cloud detectará automáticamente el cambio y **reiniciará la aplicación** con los nuevos datos cargados.

---

## 🧠 Tecnologías Usadas

- **Python 3.x**
- **Streamlit**
- **Pandas**
- **Scikit-learn** (para el modelo predictivo)
- **Matplotlib / Altair** (para visualización de datos)

---

## 📜 Licencia

Este proyecto se distribuye bajo la licencia **MIT**.  
Podés usarlo, modificarlo y compartirlo libremente.

---

✨ *Desarrollado con pasión por el análisis del comportamiento de hormigas.*
