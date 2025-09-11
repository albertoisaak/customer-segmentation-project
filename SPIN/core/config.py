"""
Configuration module for customer segmentation application.

This module contains configuration settings, constants, and utility functions
for the customer segmentation application.
"""

import os
from typing import Dict, List, Any


class Config:
    """
    Configuration class for the customer segmentation application.
    
    This class contains all configuration settings, constants, and default values
    used throughout the application.
    """
    
    # Application settings
    APP_TITLE = "Análisis de Segmentación de Clientes"
    APP_ICON = "🎯"
    PAGE_LAYOUT = "wide"
    
    # Data processing settings
    MIN_ROWS = 10
    MAX_VARIABLES = 15
    
    # ML algorithm settings
    ML_ALGORITHMS = ["K-Means", "Agglomerative", "DBSCAN"]
    DEFAULT_CLUSTERS = 5
    DEFAULT_RANDOM_STATE = 42
    MIN_CLUSTERS = 2
    MAX_CLUSTERS = 10
    
    # Gen AI settings
    AI_MODELS = ["gpt-3.5-turbo", "gpt-4"]
    DEFAULT_TEMPERATURE = 0.7
    MIN_TEMPERATURE = 0.0
    MAX_TEMPERATURE = 2.0
    
    # Variable selection keywords
    EXCLUDE_KEYWORDS = [
        'date', 'fecha', 'time', 'hora', 'timestamp', 'id', 'user_id',
        'customer_id', 'client_id', 'name', 'nombre', 'email', 'phone',
        'telefono', 'address', 'direccion', 'city', 'ciudad', 'state',
        'estado', 'country', 'pais', 'zip', 'postal', 'code', 'codigo'
    ]
    
    DEMOGRAPHIC_KEYWORDS = ['age', 'edad', 'gender', 'genero', 'income', 'ingreso']
    TRANSACTION_KEYWORDS = ['transaction', 'transaccion', 'purchase', 'compra', 'ticket', 'ticket']
    TEMPORAL_KEYWORDS = ['day', 'dia', 'month', 'mes', 'year', 'año', 'frequency', 'frecuencia']
    AMOUNT_KEYWORDS = ['amount', 'monto', 'total', 'sum', 'suma', 'avg', 'promedio', 'average']
    
    # Visualization settings
    PLOT_COLORS = {
        'ml': 'viridis',
        'ai': 'plasma'
    }
    
    # Report settings
    REPORT_SECTIONS = [
        "Información General",
        "Métricas de Calidad",
        "Distribución de Clusters",
        "Conclusión",
        "Recomendaciones"
    ]
    
    # Business questions
    BUSINESS_QUESTIONS = [
        "¿Cuántos segmentos distintos de clientes existen?",
        "¿Cuáles son las características clave de cada segmento?",
        "¿Qué comportamientos específicos tiene cada segmento?",
        "¿Qué estrategias de retención son más efectivas por segmento?",
        "¿Qué oportunidades de crecimiento identificas por segmento?",
        "¿Qué métricas de seguimiento recomiendas por segmento?"
    ]


class Utils:
    """
    Utility functions for the customer segmentation application.
    """
    
    @staticmethod
    def format_number(number: int) -> str:
        """
        Format number with thousands separator.
        
        Args:
            number: Number to format
            
        Returns:
            str: Formatted number string
        """
        return f"{number:,}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 1) -> str:
        """
        Format percentage value.
        
        Args:
            value: Percentage value (0-100)
            decimals: Number of decimal places
            
        Returns:
            str: Formatted percentage string
        """
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def get_quality_label(score: float, metric_type: str) -> str:
        """
        Get quality label based on metric score.
        
        Args:
            score: Metric score
            metric_type: Type of metric ('silhouette', 'calinski', 'davies')
            
        Returns:
            str: Quality label
        """
        if metric_type == 'silhouette':
            if score > 0.5:
                return 'Excelente'
            elif score > 0.3:
                return 'Buena'
            else:
                return 'Aceptable'
        elif metric_type == 'calinski':
            if score > 200:
                return 'Excelente separación'
            elif score > 100:
                return 'Buena separación'
            else:
                return 'Separación aceptable'
        elif metric_type == 'davies':
            if score < 0.5:
                return 'Excelente'
            elif score < 1.0:
                return 'Buena'
            else:
                return 'Aceptable'
        else:
            return 'N/A'
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """
        Validate OpenAI API key format.
        
        Args:
            api_key: API key to validate
            
        Returns:
            bool: True if valid format
        """
        if not api_key:
            return False
        
        # Basic format validation
        if not api_key.startswith('sk-'):
            return False
        
        if len(api_key) < 20:
            return False
        
        return True
    
    @staticmethod
    def get_timestamp() -> str:
        """
        Get current timestamp in formatted string.
        
        Returns:
            str: Formatted timestamp
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ErrorMessages:
    """
    Centralized error messages for the application.
    """
    
    # Data loading errors
    NO_DATA_LOADED = "❌ No hay datos cargados"
    EMPTY_DATASET = "❌ Dataset vacío"
    SMALL_DATASET = "❌ Dataset muy pequeño (mínimo 10 filas)"
    UNSUPPORTED_FORMAT = "❌ Formato de datos no soportado"
    
    # Preprocessing errors
    NO_NUMERIC_VARIABLES = "❌ No se encontraron variables numéricas válidas"
    PREPROCESSING_ERROR = "❌ Error en preprocesamiento"
    
    # ML errors
    ML_NOT_PREPROCESSED = "❌ Datos no preprocesados"
    UNSUPPORTED_ALGORITHM = "❌ Método no soportado"
    DBSCAN_NO_CLUSTERS = "❌ DBSCAN no encontró clusters válidos"
    
    # AI errors
    AI_NOT_PREPROCESSED = "❌ Datos no preprocesados"
    INVALID_API_KEY = "❌ API key inválida"
    AI_API_ERROR = "❌ Error en Gen AI"
    
    # General errors
    UNKNOWN_ERROR = "❌ Error desconocido"
    COMPARISON_ERROR = "❌ Error en análisis comparativo"


class SuccessMessages:
    """
    Centralized success messages for the application.
    """
    
    # Data loading success
    DATA_LOADED = "✅ Datos cargados exitosamente"
    DATA_PREPROCESSED = "✅ Datos preprocesados"
    
    # Analysis success
    ML_COMPLETED = "✅ Análisis ML tradicional completado"
    AI_COMPLETED = "✅ Análisis Gen AI completado"
    COMPARISON_COMPLETED = "✅ Análisis comparativo completado"
    
    # Report success
    REPORT_GENERATED = "✅ Reporte generado exitosamente"
    REPORT_DOWNLOADED = "✅ Reporte descargado exitosamente"
