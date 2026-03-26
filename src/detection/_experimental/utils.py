"""
Utility functions for vehicle data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict


def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    

def plot_signal_distribution(df: pd.DataFrame, 
                            signal_cols: List[str],
                            figsize: tuple = (15, 10)):
    """
    Plot distribution of CAN signals.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing signals
    signal_cols : List[str]
        List of signal columns to plot
    figsize : tuple
        Figure size
    """
    n_cols = 3
    n_rows = (len(signal_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, col in enumerate(signal_cols):
        if col in df.columns:
            axes[idx].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
    
    # Hide unused subplots
    for idx in range(len(signal_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_time_series(df: pd.DataFrame,
                     timestamp_col: str,
                     value_cols: List[str],
                     figsize: tuple = (15, 8)):
    """
    Plot time series of CAN signals.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with time series data
    timestamp_col : str
        Name of timestamp column
    value_cols : List[str]
        List of columns to plot
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in value_cols:
        if col in df.columns:
            ax.plot(df[timestamp_col], df[col], label=col, alpha=0.7)
    
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Value')
    ax.set_title('CAN Signal Time Series')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def plot_correlation_matrix(df: pd.DataFrame,
                           feature_cols: Optional[List[str]] = None,
                           figsize: tuple = (12, 10)):
    """
    Plot correlation matrix of features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features
    feature_cols : List[str], optional
        Specific columns to include in correlation matrix
    figsize : tuple
        Figure size
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = df[feature_cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, square=True, linewidths=1)
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    
    return fig


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics of the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    Dict : Summary statistics
    """
    summary = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'column_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    return summary


def save_dataframe(df: pd.DataFrame, filepath: Path, format: str = 'csv'):
    """
    Save dataframe to file.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to save
    filepath : Path
        Output file path
    format : str
        File format ('csv', 'parquet', 'pickle')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'parquet':
        df.to_parquet(filepath, index=False)
    elif format == 'pickle':
        df.to_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Data saved to {filepath}")
