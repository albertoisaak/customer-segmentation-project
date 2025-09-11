"""
Core analyzer module for customer segmentation.

This module contains the main SegmentationAnalyzer class that orchestrates
both traditional ML and Gen AI segmentation approaches.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from datetime import datetime

from .ml_analyzer import MLAnalyzer
from .ai_analyzer import AIAnalyzer


class SegmentationAnalyzer:
    """
    Main analyzer class for customer segmentation using both traditional ML and Gen AI.
    
    This class orchestrates the segmentation process, handling data preprocessing,
    traditional ML clustering, and Gen AI semantic analysis.
    
    Attributes:
        df (pd.DataFrame): Original dataset
        df_processed (pd.DataFrame): Preprocessed dataset
        df_clustering (pd.DataFrame): Dataset ready for clustering
        df_scaled (pd.DataFrame): Scaled dataset for ML algorithms
        ml_analyzer (MLAnalyzer): Traditional ML analyzer instance
        ai_analyzer (AIAnalyzer): Gen AI analyzer instance
    """
    
    def __init__(self):
        """Initialize the SegmentationAnalyzer with empty datasets."""
        self.df = None
        self.df_processed = None
        self.df_clustering = None
        self.df_scaled = None
        
        # Initialize specialized analyzers
        self.ml_analyzer = MLAnalyzer()
        self.ai_analyzer = AIAnalyzer()
    
    def load_data(self, data_source) -> Tuple[bool, str]:
        """
        Load data from various sources (CSV file, DataFrame, etc.).
        
        Args:
            data_source: Data source (file uploader object, DataFrame, or file path)
            
        Returns:
            Tuple[bool, str]: Success status and message
        """
        try:
            if hasattr(data_source, 'read'):  # File uploader object
                self.df = pd.read_csv(data_source)
            elif isinstance(data_source, pd.DataFrame):  # Direct DataFrame
                self.df = data_source.copy()
            elif isinstance(data_source, str):  # File path
                self.df = pd.read_csv(data_source)
            else:
                return False, "❌ Formato de datos no soportado"
            
            # Basic data validation
            if self.df.empty:
                return False, "❌ Dataset vacío"
            
            if self.df.shape[0] < 10:
                return False, "❌ Dataset muy pequeño (mínimo 10 filas)"
            
            return True, f"✅ Datos cargados exitosamente: {self.df.shape[0]:,} filas, {self.df.shape[1]} columnas"
            
        except Exception as e:
            return False, f"❌ Error cargando datos: {str(e)}"
    
    def preprocess_data(self, selected_vars: Optional[list] = None) -> Tuple[bool, str]:
        """
        Preprocess data for segmentation analysis.
        
        Args:
            selected_vars: List of variables to use for clustering (auto-selected if None)
            
        Returns:
            Tuple[bool, str]: Success status and message
        """
        try:
            if self.df is None:
                return False, "❌ No hay datos cargados"
            
            # Create processed copy
            self.df_processed = self.df.copy()
            
            # Auto-select variables if not provided
            if selected_vars is None:
                selected_vars = self._auto_select_variables()
            
            # Filter to available numeric variables
            available_vars = self._filter_numeric_variables(selected_vars)
            
            if len(available_vars) == 0:
                return False, "❌ No se encontraron variables numéricas válidas"
            
            # Create clustering dataset
            self.df_clustering = self.df_processed[available_vars].copy()
            
            # Handle missing values
            self.df_clustering = self.df_clustering.fillna(0)
            
            # Scale data for ML algorithms
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            self.df_scaled = pd.DataFrame(
                scaler.fit_transform(self.df_clustering),
                columns=self.df_clustering.columns,
                index=self.df_clustering.index
            )
            
            return True, f"✅ Datos preprocesados: {len(available_vars)} variables seleccionadas automáticamente"
            
        except Exception as e:
            return False, f"❌ Error en preprocesamiento: {str(e)}"
    
    def _auto_select_variables(self) -> list:
        """
        Automatically select variables based on column name patterns.
        
        Returns:
            list: List of selected variable names
        """
        all_cols = self.df_processed.columns.tolist()
        selected_vars = []
        
        # Keywords to exclude
        exclude_keywords = [
            'date', 'fecha', 'time', 'hora', 'timestamp', 'id', 'user_id',
            'customer_id', 'client_id', 'name', 'nombre', 'email', 'phone',
            'telefono', 'address', 'direccion', 'city', 'ciudad', 'state',
            'estado', 'country', 'pais', 'zip', 'postal', 'code', 'codigo'
        ]
        
        # Demographic keywords
        demographic_keywords = ['age', 'edad', 'gender', 'genero', 'income', 'ingreso']
        for col in all_cols:
            if (any(keyword in col.lower() for keyword in demographic_keywords) and
                not any(exclude in col.lower() for exclude in exclude_keywords)):
                selected_vars.append(col)
        
        # Transaction keywords
        transaction_keywords = ['transaction', 'transaccion', 'purchase', 'compra', 'ticket', 'ticket']
        for col in all_cols:
            if (any(keyword in col.lower() for keyword in transaction_keywords) and
                not any(exclude in col.lower() for exclude in exclude_keywords)):
                selected_vars.append(col)
        
        # Temporal keywords
        temporal_keywords = ['day', 'dia', 'month', 'mes', 'year', 'año', 'frequency', 'frecuencia']
        for col in all_cols:
            if (any(keyword in col.lower() for keyword in temporal_keywords) and
                not any(exclude in col.lower() for exclude in exclude_keywords)):
                selected_vars.append(col)
        
        # Amount keywords
        amount_keywords = ['amount', 'monto', 'total', 'sum', 'suma', 'avg', 'promedio', 'average']
        for col in all_cols:
            if (any(keyword in col.lower() for keyword in amount_keywords) and
                not any(exclude in col.lower() for exclude in exclude_keywords)):
                selected_vars.append(col)
        
        # Remove duplicates and limit to 15 variables
        selected_vars = list(set(selected_vars))
        if len(selected_vars) > 15:
            selected_vars = selected_vars[:15]
        
        return selected_vars
    
    def _filter_numeric_variables(self, variables: list) -> list:
        """
        Filter variables to only include numeric or convertible ones.
        
        Args:
            variables: List of variable names to check
            
        Returns:
            list: List of valid numeric variable names
        """
        available_vars = []
        
        for var in variables:
            if var in self.df_processed.columns:
                if self._is_numeric_column(var):
                    available_vars.append(var)
                else:
                    try:
                        # Try to convert to numeric
                        self.df_processed[var] = pd.to_numeric(self.df_processed[var], errors='coerce')
                        if not self.df_processed[var].isna().all():
                            available_vars.append(var)
                    except:
                        continue
        
        return available_vars
    
    def _is_numeric_column(self, column_name: str) -> bool:
        """
        Check if a column is numeric or can be safely converted.
        
        Args:
            column_name: Name of the column to check
            
        Returns:
            bool: True if column is numeric
        """
        try:
            if pd.api.types.is_numeric_dtype(self.df_processed[column_name]):
                return True
            
            # Check sample values
            sample_values = self.df_processed[column_name].dropna().head(10)
            if len(sample_values) == 0:
                return False
            
            pd.to_numeric(sample_values, errors='raise')
            return True
        except:
            return False
    
    def run_traditional_ml(self, method: str, n_clusters: int, random_state: int) -> Tuple[bool, Any]:
        """
        Run traditional ML clustering analysis.
        
        Args:
            method: Clustering method ('K-Means', 'Agglomerative', 'DBSCAN')
            n_clusters: Number of clusters
            random_state: Random state for reproducibility
            
        Returns:
            Tuple[bool, Any]: Success status and results
        """
        if not hasattr(self, 'df_scaled') or self.df_scaled is None:
            return False, "❌ Datos no preprocesados"
        
        return self.ml_analyzer.run_clustering(
            self.df_scaled, method, n_clusters, random_state
        )
    
    def run_gen_ai(self, api_key: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7) -> Tuple[bool, Any]:
        """
        Run Gen AI semantic segmentation analysis.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            temperature: Model temperature for creativity
            
        Returns:
            Tuple[bool, Any]: Success status and results
        """
        if not hasattr(self, 'df_scaled') or self.df_scaled is None:
            return False, "❌ Datos no preprocesados"
        
        return self.ai_analyzer.run_semantic_analysis(
            self.df, self.df_clustering, api_key, model, temperature
        )
    
    def generate_comparison_analysis(self, ml_result: Dict, ai_result: Dict, api_key: str) -> Tuple[bool, Any]:
        """
        Generate AI-powered comparison analysis between ML and Gen AI results.
        
        Args:
            ml_result: Traditional ML results
            ai_result: Gen AI results
            api_key: OpenAI API key
            
        Returns:
            Tuple[bool, Any]: Success status and comparison analysis
        """
        return self.ai_analyzer.generate_comparison_analysis(ml_result, ai_result, api_key)
