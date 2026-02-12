import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from datetime import datetime, timedelta


class LightGBMModel:
    def __init__(self, horizon=7):
        self.horizon = horizon
        self.models = []
        self.scaler = StandardScaler()
        
    def create_features(self, data, lookback=60):
        """Create technical indicators and lag features"""
        features = []
        
        for i in range(len(data) - lookback - self.horizon + 1):
            window = data[i:i + lookback]
            
            # Basic statistical features
            feat = [
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                window[-1],  # Last value
                window[0],   # First value
            ]
            
            # Technical indicators
            # Moving averages
            feat.extend([
                np.mean(window[-5:]),   # 5-day MA
                np.mean(window[-10:]),  # 10-day MA
                np.mean(window[-20:]),  # 20-day MA
            ])
            
            # Momentum indicators
            if len(window) >= 5:
                feat.append((window[-1] - window[-5]) / window[-5])  # 5-day return
            else:
                feat.append(0)
                
            if len(window) >= 10:
                feat.append((window[-1] - window[-10]) / window[-10])  # 10-day return
            else:
                feat.append(0)
            
            # Volatility
            feat.append(np.std(np.diff(window)))
            
            # Trend indicators
            feat.append(window[-1] - window[0])  # Price change over window
            
            features.append(feat)
            
        return np.array(features)
    
    def prepare_data(self, series, lookback=60):
        """Prepare features and targets for LightGBM"""
        # Create features
        X = self.create_features(series, lookback)
        
        # Create targets (next horizon days)
        y = []
        for i in range(len(series) - lookback - self.horizon + 1):
            targets = series[i + lookback:i + lookback + self.horizon]
            y.append(targets)
        
        return X, np.array(y)
    
    def fit(self, series, lookback=60):
        """Train LightGBM models for each horizon step"""
        X, y = self.prepare_data(series, lookback)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train separate model for each horizon step
        self.models = []
        for i in range(self.horizon):
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
            model.fit(X_scaled, y[:, i])
            self.models.append(model)
    
    def predict(self, series, lookback=60):
        """Make predictions for future horizon"""
        if not self.models:
            raise ValueError("Model not trained yet")
        
        # Create features from the last window
        X = self.create_features(series, lookback)
        if len(X) == 0:
            # Fallback to simple prediction if not enough data
            last_value = series[-1]
            return np.array([last_value] * self.horizon)
        
        X_scaled = self.scaler.transform(X[-1:])  # Use last window
        
        # Predict each horizon step
        predictions = []
        for model in self.models:
            pred = model.predict(X_scaled)[0]
            predictions.append(pred)
        
        return np.array(predictions)


class StackingEnsemble:
    def __init__(self, horizon=7):
        self.horizon = horizon
        self.base_models = {}
        self.meta_models = []
        self.scaler = StandardScaler()
        
    def create_base_predictions(self, series, lookback=60):
        """Generate predictions from base models"""
        from .data_prep import run_arima, run_sarima, run_prophet
        from .deep_models import LSTMModel, GRUModel, TransformerModel
        from .train import train_model
        
        # Prepare data
        train_size = int(len(series) * 0.8)
        train_series = series[:train_size]
        
        # Classical model predictions
        try:
            arima_pred = run_arima(train_series, steps=self.horizon)
        except:
            arima_pred = np.array([train_series[-1]] * self.horizon)
            
        try:
            sarima_pred = run_sarima(train_series, steps=self.horizon)
        except:
            sarima_pred = np.array([train_series[-1]] * self.horizon)
        
        # Deep learning models (simplified for ensemble)
        lstm_pred = np.array([train_series[-1]] * self.horizon)
        gru_pred = np.array([train_series[-1]] * self.horizon)
        transformer_pred = np.array([train_series[-1]] * self.horizon)
        
        # LightGBM predictions
        lgb_model = LightGBMModel(self.horizon)
        try:
            lgb_model.fit(train_series, lookback)
            lgb_pred = lgb_model.predict(train_series, lookback)
        except:
            lgb_pred = np.array([train_series[-1]] * self.horizon)
        
        return {
            'arima': arima_pred,
            'sarima': sarima_pred,
            'lstm': lstm_pred,
            'gru': gru_pred,
            'transformer': transformer_pred,
            'lightgbm': lgb_pred
        }
    
    def fit(self, series, lookback=60):
        """Train the stacking ensemble"""
        # Get base model predictions
        base_preds = self.create_base_predictions(series, lookback)
        
        # Create meta-features
        meta_features = []
        for i in range(self.horizon):
            features = [
                base_preds['arima'][i],
                base_preds['sarima'][i],
                base_preds['lstm'][i],
                base_preds['gru'][i],
                base_preds['transformer'][i],
                base_preds['lightgbm'][i]
            ]
            meta_features.append(features)
        
        meta_features = np.array(meta_features)
        
        # Scale meta-features
        meta_features_scaled = self.scaler.fit_transform(meta_features)
        
        # Train meta-models (one for each horizon step)
        self.meta_models = []
        for i in range(self.horizon):
            # Use Ridge regression as meta-model
            meta_model = Ridge(alpha=1.0)
            # Simple target: weighted average of base predictions
            target = np.mean([
                base_preds['arima'][i],
                base_preds['sarima'][i],
                base_preds['lightgbm'][i]
            ])
            meta_model.fit(meta_features_scaled[i:i+1], [target])
            self.meta_models.append(meta_model)
    
    def predict(self, series, lookback=60):
        """Make ensemble predictions"""
        if not self.meta_models:
            raise ValueError("Ensemble not trained yet")
        
        # Get base model predictions
        base_preds = self.create_base_predictions(series, lookback)
        
        # Create meta-features
        meta_features = []
        for i in range(self.horizon):
            features = [
                base_preds['arima'][i],
                base_preds['sarima'][i],
                base_preds['lstm'][i],
                base_preds['gru'][i],
                base_preds['transformer'][i],
                base_preds['lightgbm'][i]
            ]
            meta_features.append(features)
        
        meta_features = np.array(meta_features)
        meta_features_scaled = self.scaler.transform(meta_features)
        
        # Make final predictions
        predictions = []
        for i, meta_model in enumerate(self.meta_models):
            pred = meta_model.predict(meta_features_scaled[i:i+1])[0]
            predictions.append(pred)
        
        return np.array(predictions)


# Utility function to run LightGBM model
def run_lightgbm(series, horizon=7, lookback=60):
    """Convenience function to run LightGBM model"""
    model = LightGBMModel(horizon)
    model.fit(series, lookback)
    return model.predict(series, lookback)


# Utility function to run Stacking Ensemble
def run_stacking_ensemble(series, horizon=7, lookback=60):
    """Convenience function to run Stacking Ensemble"""
    ensemble = StackingEnsemble(horizon)
    ensemble.fit(series, lookback)
    return ensemble.predict(series, lookback)
