"""
Streamlit application for customer segmentation analysis.

This is the main Streamlit application that provides a user interface for
both traditional ML and Gen AI customer segmentation approaches.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Import our refactored modules
from core import SegmentationAnalyzer


def main():
    """
    Main Streamlit application function.
    
    This function sets up the Streamlit interface and handles the main
    application flow for customer segmentation analysis.
    """
    # Page configuration
    st.set_page_config(
        page_title="Análisis de Segmentación de Clientes",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🎯 Análisis de Segmentación de Clientes")
    st.markdown("""
    **Análisis avanzado de segmentación usando Machine Learning tradicional e Inteligencia Artificial Generativa**
    
    - 🔬 **ML Tradicional**: Algoritmos matemáticos (K-Means, Agglomerative, DBSCAN)
    - 🧠 **Gen AI**: Segmentación semántica basada en patrones de comportamiento
    """)
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SegmentationAnalyzer()
    
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    # Sidebar configuration
    setup_sidebar()
    
    # Main content
    setup_main_content()
    
    # Results display
    display_results()


def setup_sidebar():
    """
    Setup the sidebar with configuration options.
    """
    st.sidebar.title("⚙️ Configuración")
    
    # Data upload section
    st.sidebar.subheader("📁 Cargar Datos")
    uploaded_file = st.sidebar.file_uploader(
        "Selecciona un archivo CSV",
        type=['csv'],
        help="Sube tu archivo de datos para segmentación"
    )
    
    if uploaded_file is not None:
        success, message = st.session_state.analyzer.load_data(uploaded_file)
        if success:
            st.sidebar.success(message)
            
            # Show dataset information
            st.sidebar.subheader("📊 Información del Dataset")
            st.sidebar.write(f"**Filas:** {st.session_state.analyzer.df.shape[0]:,}")
            st.sidebar.write(f"**Columnas:** {st.session_state.analyzer.df.shape[1]}")
            st.sidebar.write(f"**Valores faltantes:** {st.session_state.analyzer.df.isnull().sum().sum():,}")
            
            # Automatic preprocessing
            st.sidebar.subheader("🔧 Preprocesamiento Automático")
            st.sidebar.info("💡 Los datos se preprocesan automáticamente al cargar el archivo.")
            
            with st.spinner("Preprocesando datos automáticamente..."):
                success, message = st.session_state.analyzer.preprocess_data()
                if success:
                    st.sidebar.success(message)
                    # Show selected variables
                    if hasattr(st.session_state.analyzer, 'df_clustering'):
                        st.sidebar.write("**Variables seleccionadas automáticamente:**")
                        for i, var in enumerate(st.session_state.analyzer.df_clustering.columns, 1):
                            st.sidebar.write(f"{i}. {var}")
                    else:
                        st.sidebar.error("❌ Error en el preprocesamiento")
                else:
                    st.sidebar.error(f"❌ Error en preprocesamiento: {message}")
        else:
            st.sidebar.error(message)
    
    # Method configuration
    st.sidebar.subheader("🤖 Métodos de Análisis")
    
    # Traditional ML configuration
    st.sidebar.subheader("🔬 Machine Learning Tradicional")
    st.session_state.ml_method = st.sidebar.selectbox(
        "Algoritmo",
        ["K-Means", "Agglomerative", "DBSCAN"],
        help="Algoritmo de clustering a utilizar"
    )
    
    st.session_state.ml_clusters = st.sidebar.slider(
        "Número de clusters",
        min_value=2,
        max_value=10,
        value=5,
        help="Número de clusters para el algoritmo"
    )
    
    st.session_state.ml_random_state = st.sidebar.number_input(
        "Random State",
        min_value=0,
        max_value=1000,
        value=42,
        help="Semilla aleatoria para reproducibilidad"
    )
    
    # Gen AI configuration
    st.sidebar.subheader("🧠 Inteligencia Artificial Generativa")
    st.session_state.api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Ingresa tu API key de OpenAI"
    )
    
    st.session_state.ai_model = st.sidebar.selectbox(
        "Modelo de IA",
        ["gpt-3.5-turbo", "gpt-4"],
        help="Modelo de OpenAI a utilizar"
    )
    
    st.sidebar.info("🧠 Gen AI usa **Segmentación Semántica** - No requiere parámetros tradicionales")
    
    st.session_state.temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Creatividad del modelo (0=determinístico, 2=muy creativo)"
    )


def setup_main_content():
    """
    Setup the main content area with analysis buttons.
    """
    # Main content columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔬 Machine Learning Tradicional")
        
        if st.button("▶️ Ejecutar ML Tradicional", type="primary"):
            if hasattr(st.session_state.analyzer, 'df_scaled') and st.session_state.analyzer.df_scaled is not None:
                with st.spinner("Ejecutando análisis ML tradicional..."):
                    success, result = st.session_state.analyzer.run_traditional_ml(
                        st.session_state.ml_method, st.session_state.ml_clusters, st.session_state.ml_random_state
                    )
                    
                    if success:
                        st.session_state.results["ML Tradicional"] = result
                        st.success("✅ Análisis ML tradicional completado")
                    else:
                        st.error(f"❌ Error en ML tradicional: {result}")
            else:
                st.error("❌ Datos no preprocesados")
    
    with col2:
        st.subheader("🧠 Inteligencia Artificial Generativa")
        
        if st.button("▶️ Ejecutar Gen AI", type="primary"):
            if hasattr(st.session_state.analyzer, 'df_scaled') and st.session_state.analyzer.df_scaled is not None and st.session_state.api_key:
                with st.spinner("Ejecutando análisis con IA..."):
                    success, result = st.session_state.analyzer.run_gen_ai(
                        st.session_state.api_key, st.session_state.ai_model, st.session_state.temperature
                    )
                    
                    if success:
                        st.session_state.results["Gen AI"] = result
                        st.success("✅ Análisis Gen AI completado")
                    else:
                        st.error(f"❌ Error en Gen AI: {result}")
            else:
                st.error("❌ Datos no preprocesados o API key faltante")


def display_results():
    """
    Display analysis results for both ML and Gen AI approaches.
    """
    # Traditional ML Results
    if "ML Tradicional" in st.session_state.results:
        display_ml_results()
    
    # Gen AI Results
    if "Gen AI" in st.session_state.results:
        display_ai_results()
    
    # Comparison Analysis eliminada por solicitud del usuario


def display_ml_results():
    """
    Display traditional ML analysis results.
    """
    st.header("🔬 Resultados - Machine Learning Tradicional")
    
    ml_result = st.session_state.results["ML Tradicional"]
    
    # Metrics display
    st.subheader("📈 Métricas del Modelo")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Silhouette Score", f"{ml_result['metrics']['silhouette_score']:.3f}")
    with col2:
        st.metric("Calinski-Harabasz", f"{ml_result['metrics']['calinski_harabasz_score']:.1f}")
    with col3:
        st.metric("Davies-Bouldin", f"{ml_result['metrics']['davies_bouldin_score']:.3f}")
    
    # Cluster distribution
    st.subheader("📊 Distribución de Clusters")
    cluster_dist = ml_result["cluster_distribution"]
    fig_ml = px.bar(
        x=list(cluster_dist.keys()),
        y=list(cluster_dist.values()),
        title="Usuarios por Cluster - ML Tradicional",
        labels={'x': 'Cluster', 'y': 'Número de Usuarios'},
        color=list(cluster_dist.values()),
        color_continuous_scale="viridis"
    )
    st.plotly_chart(fig_ml, use_container_width=True)
    
    # Report generation
    if st.button("📄 Generar Reporte ML Tradicional"):
        report = st.session_state.analyzer.ml_analyzer.generate_ml_report(
            ml_result, st.session_state.analyzer.df
        )
        st.markdown(report)


def display_ai_results():
    """
    Display Gen AI analysis results.
    """
    st.header("🤖 Resultados - Inteligencia Artificial Generativa")
    
    ai_result = st.session_state.results["Gen AI"]
    
    # AI information
    st.subheader("🧠 Segmentación Semántica")
    st.info("Gen AI genera segmentación basada en análisis de patrones de comportamiento, no clusters tradicionales.")
    st.info(f"**Tipo de segmentación**: {ai_result.get('segmentation_type', 'Semantic Analysis')}")
    st.info(f"**Clusters generados**: {ai_result.get('clusters_generated', 'Automático')}")
    
    # AI model information
    col_ai1, col_ai2, col_ai3 = st.columns(3)
    with col_ai1:
        st.metric("Modelo IA", ai_result.get('api_model', 'N/A'))
    with col_ai2:
        st.metric("Temperature", ai_result.get('temperature', 'N/A'))
    with col_ai3:
        st.metric("Independiente", "✅ Sí" if ai_result.get("is_independent", False) else "❌ No")
    
    # AI Analysis
    st.subheader("🧠 Análisis Generado por IA")
    st.markdown(ai_result["ai_analysis"])
    
    # AI Insights
    st.subheader("💡 Insights Adicionales de IA")
    st.markdown(ai_result["ai_insights"])
    
    # Report generation
    if st.button("📄 Generar Reporte Gen AI"):
        report = st.session_state.analyzer.ai_analyzer.generate_ai_report(
            ai_result, st.session_state.analyzer.df
        )
        st.markdown(report)


# Función de análisis comparativo eliminada por solicitud del usuario


if __name__ == "__main__":
    main()
