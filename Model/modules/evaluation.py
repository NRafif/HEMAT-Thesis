"""
Model evaluation and comparison module.

This module provides tools for evaluating individual models and
comparing multiple models across different metrics and visualizations.

Classes:
    - ModelEvaluator: Evaluate single model performance
    - ComparativeAnalyzer: Compare multiple models
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class ModelEvaluator:
    """
    Evaluate single model performance with multiple metrics and visualizations.
    
    Provides comprehensive evaluation including:
    - Performance metrics (MAE, RMSE, R²)
    - Prediction visualizations
    - Residual analysis
    """
    
    def __init__(self, model_name: str = 'Model'):
        """
        Initialize evaluator.
        
        Args:
            model_name: Name of the model being evaluated
        """
        self.model_name = model_name
        self.metrics = {}
        self.predictions = None
        self.actuals = None
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        return_dict: bool = True
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            return_dict: If True, return dict; else print results
        
        Returns:
            Dictionary with MAE, RMSE, and R² scores
        """
        # Store for later use
        self.actuals = y_true
        self.predictions = y_pred
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        self.metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        if not return_dict:
            print(f"\n{'='*50}")
            print(f"Evaluation Results: {self.model_name}")
            print(f"{'='*50}")
            print(f"MAE  (Mean Absolute Error)    : {mae:.4f}")
            print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
            print(f"R²   (R-squared)              : {r2:.4f}")
            print(f"{'='*50}\n")
        
        return self.metrics
    
    def plot_predictions(
        self,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: Actual values (uses stored if None)
            y_pred: Predicted values (uses stored if None)
            title: Plot title
            save_path: Path to save figure
            show: Whether to display plot
        
        Returns:
            Matplotlib figure object
        """
        # Use stored values if not provided
        if y_true is None:
            y_true = self.actuals
        if y_pred is None:
            y_pred = self.predictions
        
        if y_true is None or y_pred is None:
            raise ValueError("No predictions available. Run evaluate() first or provide y_true and y_pred.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Labels and title
        ax.set_xlabel('Actual Values (kW)', fontsize=12)
        ax.set_ylabel('Predicted Values (kW)', fontsize=12)
        
        if title is None:
            title = f'{self.model_name}: Actual vs Predicted'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add metrics to plot
        if self.metrics:
            metrics_text = f"R² = {self.metrics['R2']:.4f}\nRMSE = {self.metrics['RMSE']:.4f}\nMAE = {self.metrics['MAE']:.4f}"
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_residuals(
        self,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot residual analysis (residuals vs predicted values).
        
        Args:
            y_true: Actual values (uses stored if None)
            y_pred: Predicted values (uses stored if None)
            title: Plot title
            save_path: Path to save figure
            show: Whether to display plot
        
        Returns:
            Matplotlib figure object
        """
        # Use stored values if not provided
        if y_true is None:
            y_true = self.actuals
        if y_pred is None:
            y_pred = self.predictions
        
        if y_true is None or y_pred is None:
            raise ValueError("No predictions available. Run evaluate() first or provide y_true and y_pred.")
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', lw=2)
        ax1.set_xlabel('Predicted Values (kW)', fontsize=12)
        ax1.set_ylabel('Residuals (kW)', fontsize=12)
        ax1.set_title('Residual Plot', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Residual distribution
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Residuals (kW)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def analyze_residuals(
        self,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Analyze residual statistics.
        
        Args:
            y_true: Actual values (uses stored if None)
            y_pred: Predicted values (uses stored if None)
        
        Returns:
            Dictionary with residual statistics
        """
        # Use stored values if not provided
        if y_true is None:
            y_true = self.actuals
        if y_pred is None:
            y_pred = self.predictions
        
        if y_true is None or y_pred is None:
            raise ValueError("No predictions available. Run evaluate() first.")
        
        residuals = y_true - y_pred
        
        stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'median': np.median(residuals),
            'q25': np.percentile(residuals, 25),
            'q75': np.percentile(residuals, 75)
        }
        
        return stats


class ComparativeAnalyzer:
    """
    Compare multiple models across metrics and visualizations.
    
    Stores results from multiple models and provides comparison tools:
    - Comparison tables
    - Metric comparison plots
    - Feature importance comparison
    """
    
    def __init__(self):
        """Initialize comparative analyzer."""
        self.results = {}
    
    def add_model_results(
        self,
        model_name: str,
        metrics: Dict[str, float],
        feature_importance: Optional[pd.DataFrame] = None,
        training_time: Optional[float] = None,
        predictions: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> None:
        """
        Add results from a model.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary with MAE, RMSE, R²
            feature_importance: DataFrame with feature importance
            training_time: Training time in seconds
            predictions: Tuple of (y_true, y_pred)
        """
        self.results[model_name] = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'training_time': training_time,
            'predictions': predictions
        }
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate comparison table for all models.
        
        Returns:
            DataFrame with metrics for all models
        """
        if not self.results:
            raise ValueError("No results added. Use add_model_results() first.")
        
        data = []
        for model_name, result in self.results.items():
            row = {
                'Model': model_name,
                'MAE': result['metrics']['MAE'],
                'RMSE': result['metrics']['RMSE'],
                'R²': result['metrics']['R2']
            }
            
            if result['training_time'] is not None:
                row['Training Time (s)'] = result['training_time']
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('R²', ascending=False).reset_index(drop=True)
        
        return df
    
    def plot_metric_comparison(
        self,
        metrics: List[str] = ['MAE', 'RMSE', 'R²'],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot bar chart comparing metrics across models.
        
        Args:
            metrics: List of metrics to plot
            title: Plot title
            save_path: Path to save figure
            show: Whether to display plot
        
        Returns:
            Matplotlib figure object
        """
        if not self.results:
            raise ValueError("No results added. Use add_model_results() first.")
        
        # Prepare data
        df = self.generate_comparison_table()
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            if metric not in df.columns:
                continue
            
            ax = axes[idx]
            
            # Bar plot
            bars = ax.bar(df['Model'], df[metric], edgecolor='black', alpha=0.7)
            
            # Color bars (best = green)
            if metric == 'R²':
                best_idx = df[metric].idxmax()
            else:
                best_idx = df[metric].idxmin()
            
            bars[best_idx].set_color('green')
            bars[best_idx].set_alpha(0.9)
            
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x labels if needed
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def plot_feature_importance_comparison(
        self,
        top_n: int = 10,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot side-by-side feature importance comparison.
        
        Args:
            top_n: Number of top features to show
            title: Plot title
            save_path: Path to save figure
            show: Whether to display plot
        
        Returns:
            Matplotlib figure object
        """
        # Filter models with feature importance
        models_with_fi = {
            name: result for name, result in self.results.items()
            if result['feature_importance'] is not None
        }
        
        if not models_with_fi:
            raise ValueError("No feature importance data available.")
        
        n_models = len(models_with_fi)
        fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(models_with_fi.items()):
            ax = axes[idx]
            fi_df = result['feature_importance'].head(top_n)
            
            # Horizontal bar plot
            ax.barh(range(len(fi_df)), fi_df['Importance'], edgecolor='black', alpha=0.7)
            ax.set_yticks(range(len(fi_df)))
            ax.set_yticklabels(fi_df['Feature'])
            ax.invert_yaxis()
            ax.set_xlabel('Importance', fontsize=11)
            ax.set_title(f'{model_name}\nTop {top_n} Features', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def get_best_model(self, metric: str = 'R²') -> str:
        """
        Get name of best performing model.
        
        Args:
            metric: Metric to use for comparison ('MAE', 'RMSE', or 'R²')
        
        Returns:
            Name of best model
        """
        if not self.results:
            raise ValueError("No results added.")
        
        df = self.generate_comparison_table()
        
        if metric == 'R²':
            best_idx = df[metric].idxmax()
        else:
            best_idx = df[metric].idxmin()
        
        return df.loc[best_idx, 'Model']
    
    def generate_summary_report(self) -> str:
        """
        Generate text summary report.
        
        Returns:
            Formatted string with comparison summary
        """
        if not self.results:
            return "No results available."
        
        df = self.generate_comparison_table()
        
        report = []
        report.append("="*60)
        report.append("MODEL COMPARISON SUMMARY")
        report.append("="*60)
        report.append("")
        report.append(df.to_string(index=False))
        report.append("")
        report.append("-"*60)
        report.append("BEST MODELS:")
        report.append(f"  • Best R²   : {self.get_best_model('R²')}")
        report.append(f"  • Best MAE  : {self.get_best_model('MAE')}")
        report.append(f"  • Best RMSE : {self.get_best_model('RMSE')}")
        report.append("="*60)
        
        return "\n".join(report)
