import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuración de la página
st.set_page_config(page_title="PharmAI - Platform v1.0", layout="centered")

st.title("🧪 PharmAI - Platform v1.0")
st.subheader("Sistema de Evaluación de IA para la Predicción de Toxicidad")

# Inicialización de estado
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False

# --- PASO 1: CARGA DE DATOS ---
st.markdown("### 1. Sincronizar Base de Datos Química")
uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        columnas_num = ["Peso", "Enlaces", "LogP", "Polares"]
        
        # Procesamiento
        df = df.dropna()
        X_raw = df[columnas_num].to_numpy()
        y_raw = df["Toxicidad"].to_numpy()
        
        X_support_raw, X_test_raw, y_support_raw, y_test_raw = train_test_split(
            X_raw, y_raw, test_size=0.3, random_state=42
        )
        
        # Guardar en sesión
        st.session_state.X_all = torch.tensor(X_raw, dtype=torch.float32)
        st.session_state.prototipo_seguro = torch.tensor(X_support_raw[y_support_raw == 0], dtype=torch.float32).mean(dim=0)
        st.session_state.prototipo_toxico = torch.tensor(X_support_raw[y_support_raw == 1], dtype=torch.float32).mean(dim=0)
        st.session_state.nombres = df["Nombre"].tolist()
        st.session_state.model_ready = True
        
        st.success("¡Base de datos sincronizada correctamente!")
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")

# --- PASO 2: INPUTS ---
st.markdown("### 2. Parámetros del Nuevo Compuesto")
col1, col2 = st.columns(2)
with col1:
    peso = st.number_input("Peso molecular:", 0.0, 1.0, 0.45)
    enlaces = st.number_input("Enlaces rotables:", 0.0, 1.0, 0.20)
with col2:
    logp = st.number_input("Hidrofobicidad (LogP):", 0.0, 1.0, 0.65)
    polares = st.number_input("Densidad átomos polares:", 0.0, 1.0, 0.35)

if st.button("Ejecutar Análisis Predictivo"):
    if st.session_state.model_ready:
        nueva_mol = torch.tensor([peso, enlaces, logp, polares], dtype=torch.float32)
        
        d_seg = torch.pairwise_distance(nueva_mol.unsqueeze(0), st.session_state.prototipo_seguro.unsqueeze(0)).item()
        d_tox = torch.pairwise_distance(nueva_mol.unsqueeze(0), st.session_state.prototipo_toxico.unsqueeze(0)).item()
        
        prob = (1/d_tox) / ((1/d_seg) + (1/d_tox)) * 100
        
        st.markdown("---")
        st.subheader("Resultado del Reporte")
        st.write(f"**Probabilidad de toxicidad:** {prob:.2f}%")
        
        if prob < 50:
            st.success("CLASIFICACIÓN: RIESGO BAJO")
        else:
            st.error("CLASIFICACIÓN: RIESGO ELEVADO")
    else:
        st.warning("Por favor, sube un archivo Excel primero.")