"""
Model explainability using SHAP and LIME.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class ModelExplainer:
    """Model explainability using SHAP and LIME."""
    
    def __init__(self, model, feature_names: List[str], X_train: np.ndarray = None):
        """
        Initialize the explainer.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            X_train: Training data for background (needed for SHAP)
        """
        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        
        self._shap_explainer = None
        self._lime_explainer = None
    
    def _get_shap_explainer(self):
        """Get or create SHAP explainer."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        if self._shap_explainer is None:
            if self.X_train is not None:
                # Use a sample for faster computation
                sample_size = min(100, len(self.X_train))
                background = self.X_train[np.random.choice(len(self.X_train), sample_size, replace=False)]
            else:
                background = None
            
            # Try different explainers based on model type
            try:
                self._shap_explainer = shap.TreeExplainer(self.model)
            except:
                try:
                    self._shap_explainer = shap.KernelExplainer(
                        self.model.predict, 
                        background or np.zeros((1, len(self.feature_names)))
                    )
                except:
                    self._shap_explainer = shap.Explainer(self.model, background)
        
        return self._shap_explainer
    
    def _get_lime_explainer(self):
        """Get or create LIME explainer."""
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required. Install with: pip install lime")
        
        if self._lime_explainer is None:
            training_data = self.X_train if self.X_train is not None else np.zeros((10, len(self.feature_names)))
            
            self._lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=self.feature_names,
                mode='regression'
            )
        
        return self._lime_explainer
    
    def explain_shap(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Explain predictions using SHAP.
        
        Args:
            X: Input features to explain
            
        Returns:
            Dictionary with SHAP values and feature importance
        """
        explainer = self._get_shap_explainer()
        shap_values = explainer.shap_values(X)
        
        # Calculate mean absolute SHAP values for feature importance
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance ranking
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        return {
            'shap_values': shap_values,
            'feature_importance': importance_df.to_dict('records'),
            'top_features': importance_df.head(10)['feature'].tolist()
        }
    
    def explain_lime(self, X: np.ndarray, instance_idx: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction using LIME.
        
        Args:
            X: Input features
            instance_idx: Index of instance to explain
            
        Returns:
            Dictionary with LIME explanation
        """
        explainer = self._get_lime_explainer()
        
        # Get prediction function
        if hasattr(self.model, 'predict'):
            predict_fn = self.model.predict
        else:
            predict_fn = lambda x: self.model(x)
        
        # Explain instance
        exp = explainer.explain_instance(
            X[instance_idx],
            predict_fn,
            num_features=10
        )
        
        # Extract feature contributions
        contributions = []
        for feature, weight in exp.as_list():
            contributions.append({
                'feature': feature,
                'weight': weight
            })
        
        return {
            'instance_explanation': contributions,
            'predicted_value': exp.predicted_value,
            'local_prediction': exp.local_pred[0] if hasattr(exp, 'local_pred') else None
        }
    
    def plot_shap_summary(self, X: np.ndarray, save_path: str = None):
        """
        Plot SHAP summary plot.
        
        Args:
            X: Input features
            save_path: Optional path to save the plot
        """
        explainer = self._get_shap_explainer()
        shap_values = explainer.shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"SHAP summary plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_shap_waterfall(self, X: np.ndarray, instance_idx: int = 0, save_path: str = None):
        """
        Plot SHAP waterfall plot for a single instance.
        
        Args:
            X: Input features
            instance_idx: Index of instance to plot
            save_path: Optional path to save the plot
        """
        explainer = self._get_shap_explainer()
        shap_values = explainer(X[instance_idx:instance_idx+1])
        
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_values[0], show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"SHAP waterfall plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_feature_importance(self, X: np.ndarray, method: str = 'shap') -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            X: Input features
            method: 'shap' or 'lime'
            
        Returns:
            DataFrame with feature importance
        """
        if method == 'shap':
            result = self.explain_shap(X)
            return pd.DataFrame(result['feature_importance'])
        elif method == 'lime':
            # Aggregate LIME explanations over multiple instances
            importances = {}
            n_samples = min(50, len(X))
            
            for i in range(n_samples):
                result = self.explain_lime(X, i)
                for contrib in result['instance_explanation']:
                    feature = contrib['feature'].split()[0]  # Get just the feature name
                    if feature not in importances:
                        importances[feature] = []
                    importances[feature].append(abs(contrib['weight']))
            
            # Calculate mean importance
            importance_df = pd.DataFrame([
                {'feature': f, 'importance': np.mean(v)}
                for f, v in importances.items()
            ]).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            raise ValueError(f"Unknown method: {method}. Use 'shap' or 'lime'")


def main():
    """Example usage of model explainability."""
    print("Model Explainability Example")
    print("="*50)
    
    # Create sample data and model
    from sklearn.ensemble import RandomForestRegressor
    
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5
    
    # Train a simple model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Create explainer
    explainer = ModelExplainer(model, feature_names, X)
    
    # SHAP explanation
    if SHAP_AVAILABLE:
        print("\nSHAP Feature Importance:")
        shap_result = explainer.explain_shap(X[:50])
        for item in shap_result['feature_importance'][:5]:
            print(f"  {item['feature']}: {item['importance']:.4f}")
    else:
        print("\nSHAP not available. Install with: pip install shap")
    
    # LIME explanation
    if LIME_AVAILABLE:
        print("\nLIME Explanation (single instance):")
        lime_result = explainer.explain_lime(X, 0)
        for item in lime_result['instance_explanation'][:5]:
            print(f"  {item['feature']}: {item['weight']:.4f}")
    else:
        print("\nLIME not available. Install with: pip install lime")


if __name__ == "__main__":
    main()
