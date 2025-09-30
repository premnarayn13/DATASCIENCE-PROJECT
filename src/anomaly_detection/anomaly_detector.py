"""
Advanced Anomaly Detection for IT Support Ticket Systems

This module implements sophisticated anomaly detection algorithms to identify
unusual patterns, predict system failures, and prevent cascade incidents.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesAnomalyDetector:
    """Detects anomalies in ticket volume time series"""
    
    def __init__(self, window_size: int = 24, contamination: float = 0.1):
        self.window_size = window_size
        self.contamination = contamination
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        
    def prepare_time_series_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare time series data for anomaly detection"""
        
        # Ensure datetime column
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Create hourly aggregations
        hourly_data = df.set_index('created_at').resample('H').agg({
            'ticket_id': 'count',
            'priority': lambda x: (x == 'Critical').sum() + (x == 'Emergency').sum(),
            'escalated': 'sum',
            'resolution_time_hours': 'mean',
            'customer_satisfaction': 'mean'
        }).fillna(0)
        
        # Rename columns
        hourly_data.columns = [
            'ticket_count', 'high_priority_count', 'escalated_count',
            'avg_resolution_time', 'avg_satisfaction'
        ]
        
        # Add time-based features
        hourly_data['hour'] = hourly_data.index.hour
        hourly_data['day_of_week'] = hourly_data.index.dayofweek
        hourly_data['is_weekend'] = hourly_data.index.dayofweek.isin([5, 6]).astype(int)
        hourly_data['is_business_hours'] = ((hourly_data.index.hour >= 9) & 
                                          (hourly_data.index.hour <= 17)).astype(int)
        
        # Add rolling statistics
        for window in [6, 12, 24]:  # 6h, 12h, 24h windows
            hourly_data[f'ticket_count_rolling_mean_{window}h'] = (
                hourly_data['ticket_count'].rolling(window=window, min_periods=1).mean()
            )
            hourly_data[f'ticket_count_rolling_std_{window}h'] = (
                hourly_data['ticket_count'].rolling(window=window, min_periods=1).std().fillna(0)
            )
        
        # Add lag features
        for lag in [1, 6, 24]:  # 1h, 6h, 24h lags
            hourly_data[f'ticket_count_lag_{lag}h'] = hourly_data['ticket_count'].shift(lag).fillna(0)
        
        return hourly_data
    
    def detect_volume_anomalies(self, time_series_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in ticket volume"""
        
        logger.info("Detecting ticket volume anomalies...")
        
        # Select features for anomaly detection
        feature_cols = [
            'ticket_count', 'high_priority_count', 'escalated_count',
            'ticket_count_rolling_mean_6h', 'ticket_count_rolling_std_6h',
            'ticket_count_rolling_mean_12h', 'ticket_count_rolling_std_12h',
            'ticket_count_lag_1h', 'ticket_count_lag_6h'
        ]
        
        X = time_series_data[feature_cols].fillna(0)
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['volume'] = scaler
        
        # Multiple anomaly detection models
        models = {
            'isolation_forest': IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            ),
            'one_class_svm': OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            ),
            'lof': LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                n_jobs=-1
            )
        }
        
        anomaly_scores = {}
        anomaly_labels = {}
        
        for name, model in models.items():
            logger.info(f"Training {name} for volume anomaly detection...")
            
            if name == 'lof':
                # LOF returns labels directly
                labels = model.fit_predict(X_scaled)
                scores = model.negative_outlier_factor_
            else:
                # Other models can return both scores and labels
                model.fit(X_scaled)
                labels = model.predict(X_scaled)
                scores = model.decision_function(X_scaled)
            
            # Convert labels (-1 for anomaly, 1 for normal)
            anomaly_labels[name] = (labels == -1).astype(int)
            anomaly_scores[name] = scores
            
            self.models[f'volume_{name}'] = model
        
        # Ensemble prediction (majority voting)
        ensemble_labels = np.zeros(len(X))
        for labels in anomaly_labels.values():
            ensemble_labels += labels
        
        # Consider as anomaly if majority of models agree
        ensemble_anomalies = (ensemble_labels >= len(models) / 2).astype(int)
        
        # Add results to dataframe
        time_series_data['volume_anomaly'] = ensemble_anomalies
        time_series_data['anomaly_score_if'] = anomaly_scores['isolation_forest']
        time_series_data['anomaly_score_svm'] = anomaly_scores['one_class_svm']
        time_series_data['anomaly_score_lof'] = anomaly_scores['lof']
        
        anomaly_summary = {
            'total_anomalies': ensemble_anomalies.sum(),
            'anomaly_rate': ensemble_anomalies.mean(),
            'anomaly_timestamps': time_series_data[ensemble_anomalies == 1].index.tolist()
        }
        
        logger.info(f"Volume anomaly detection complete. Found {anomaly_summary['total_anomalies']} anomalies")
        
        return anomaly_summary

class TicketAnomalyDetector:
    """Detects anomalies in individual ticket characteristics"""
    
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_ticket_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ticket-level anomaly detection"""
        
        # Select numerical features
        numerical_features = [
            'resolution_time_hours', 'customer_satisfaction', 'text_length',
            'word_count', 'sentence_count', 'urgency_score', 'frustration_score',
            'escalation_risk', 'user_ticket_count', 'user_avg_resolution_time'
        ]
        
        # Select categorical features to encode
        categorical_features = ['priority', 'category', 'department', 'user_role']
        
        feature_df = df.copy()
        
        # Handle missing values
        for col in numerical_features:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())
        
        # Encode categorical features
        for col in categorical_features:
            if col in feature_df.columns:
                feature_df[f'{col}_encoded'] = feature_df[col].astype('category').cat.codes
        
        # Create interaction features
        if 'resolution_time_hours' in feature_df.columns and 'customer_satisfaction' in feature_df.columns:
            feature_df['time_satisfaction_ratio'] = (
                feature_df['resolution_time_hours'] / (feature_df['customer_satisfaction'] + 0.1)
            )
        
        if 'urgency_score' in feature_df.columns and 'frustration_score' in feature_df.columns:
            feature_df['emotional_intensity'] = (
                feature_df['urgency_score'] + feature_df['frustration_score']
            )
        
        return feature_df
    
    def detect_resolution_time_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in resolution times"""
        
        logger.info("Detecting resolution time anomalies...")
        
        # Features for resolution time anomaly detection
        feature_cols = [
            'priority_encoded', 'category_encoded', 'department_encoded',
            'text_length', 'word_count', 'urgency_score', 'frustration_score',
            'user_avg_resolution_time', 'time_satisfaction_ratio'
        ]
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) < 3:
            logger.warning("Insufficient features for resolution time anomaly detection")
            return {}
        
        X = df[available_cols].fillna(0)
        y = df['resolution_time_hours'].fillna(df['resolution_time_hours'].median())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['resolution_time'] = scaler
        
        # Use Isolation Forest for anomaly detection
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        
        # Combine features with target for anomaly detection
        X_combined = np.column_stack([X_scaled, y.values.reshape(-1, 1)])
        
        # Fit and predict
        anomaly_labels = iso_forest.fit_predict(X_combined)
        anomaly_scores = iso_forest.decision_function(X_combined)
        
        # Convert to binary labels
        anomalies = (anomaly_labels == -1).astype(int)
        
        self.models['resolution_time'] = iso_forest
        
        # Analyze anomalies
        anomaly_indices = np.where(anomalies == 1)[0]
        normal_indices = np.where(anomalies == 0)[0]
        
        anomaly_stats = {
            'total_anomalies': anomalies.sum(),
            'anomaly_rate': anomalies.mean(),
            'avg_resolution_time_anomalies': y.iloc[anomaly_indices].mean() if len(anomaly_indices) > 0 else 0,
            'avg_resolution_time_normal': y.iloc[normal_indices].mean() if len(normal_indices) > 0 else 0,
            'anomaly_tickets': df.iloc[anomaly_indices]['ticket_id'].tolist() if len(anomaly_indices) > 0 else []
        }
        
        # Add anomaly indicators to dataframe
        df['resolution_time_anomaly'] = anomalies
        df['resolution_time_anomaly_score'] = anomaly_scores
        
        logger.info(f"Resolution time anomaly detection complete. Found {anomaly_stats['total_anomalies']} anomalies")
        
        return anomaly_stats
    
    def detect_behavior_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in user/system behavior patterns"""
        
        logger.info("Detecting user behavior anomalies...")
        
        # Group by user and calculate behavior metrics
        user_behavior = df.groupby('user_id').agg({
            'ticket_id': 'count',
            'resolution_time_hours': ['mean', 'std'],
            'customer_satisfaction': 'mean',
            'escalated': 'mean',
            'urgency_score': 'mean',
            'frustration_score': 'mean'
        }).fillna(0)
        
        # Flatten column names
        user_behavior.columns = ['_'.join(col).strip() if col[1] else col[0] 
                               for col in user_behavior.columns]
        
        # Filter users with minimum activity
        min_tickets = 3
        active_users = user_behavior[user_behavior['ticket_id_count'] >= min_tickets]
        
        if len(active_users) < 10:
            logger.warning("Insufficient data for user behavior anomaly detection")
            return {}
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(active_users)
        self.scalers['user_behavior'] = scaler
        
        # Detect anomalies
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.decision_function(X_scaled)
        
        anomalies = (anomaly_labels == -1).astype(int)
        
        self.models['user_behavior'] = iso_forest
        
        # Add results
        active_users['behavior_anomaly'] = anomalies
        active_users['behavior_anomaly_score'] = anomaly_scores
        
        anomaly_users = active_users[active_users['behavior_anomaly'] == 1]
        
        behavior_stats = {
            'total_anomalous_users': anomalies.sum(),
            'anomaly_rate': anomalies.mean(),
            'anomalous_user_ids': anomaly_users.index.tolist(),
            'avg_tickets_anomalous_users': anomaly_users['ticket_id_count'].mean() if len(anomaly_users) > 0 else 0
        }
        
        logger.info(f"User behavior anomaly detection complete. Found {behavior_stats['total_anomalous_users']} anomalous users")
        
        return behavior_stats

class AutoencoderAnomalyDetector:
    """Neural network-based autoencoder for anomaly detection"""
    
    def __init__(self, encoding_dim: int = 32, contamination: float = 0.05):
        self.encoding_dim = encoding_dim
        self.contamination = contamination
        self.autoencoder = None
        self.encoder = None
        self.scaler = None
        self.threshold = None
        
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Autoencoder functionality disabled.")
    
    def build_autoencoder(self, input_dim: int) -> None:
        """Build autoencoder model"""
        
        if not TF_AVAILABLE:
            return
        
        # Input layer
        input_layer = keras.Input(shape=(input_dim,))
        
        # Encoder
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Models
        self.autoencoder = keras.Model(input_layer, decoded)
        self.encoder = keras.Model(input_layer, encoded)
        
        # Compile
        self.autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
    
    def fit(self, X: np.ndarray, validation_split: float = 0.2, epochs: int = 100) -> None:
        """Train the autoencoder"""
        
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping autoencoder training.")
            return
        
        # Scale data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model
        self.build_autoencoder(X.shape[1])
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        logger.info("Training autoencoder...")
        history = self.autoencoder.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Calculate reconstruction errors on training data
        X_pred = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
        
        # Set threshold based on contamination rate
        self.threshold = np.percentile(mse, (1 - self.contamination) * 100)
        
        logger.info(f"Autoencoder training complete. Threshold: {self.threshold:.4f}")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies using trained autoencoder"""
        
        if not TF_AVAILABLE or self.autoencoder is None:
            logger.warning("Autoencoder not available or not trained.")
            return np.array([]), np.array([])
        
        # Scale data
        X_scaled = self.scaler.transform(X)
        
        # Get reconstructions
        X_pred = self.autoencoder.predict(X_scaled)
        
        # Calculate reconstruction errors
        mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
        
        # Classify as anomaly if error > threshold
        anomalies = (mse > self.threshold).astype(int)
        
        return anomalies, mse

class CascadePredictor:
    """Predicts potential cascade failures in IT systems"""
    
    def __init__(self):
        self.system_dependencies = {}
        self.cascade_models = {}
        
    def build_dependency_graph(self, systems_df: pd.DataFrame, tickets_df: pd.DataFrame) -> None:
        """Build system dependency graph from data"""
        
        logger.info("Building system dependency graph...")
        
        # Parse system dependencies
        for _, system in systems_df.iterrows():
            system_id = system['system_id']
            dependencies = []
            
            if pd.notnull(system.get('dependencies')):
                deps = str(system['dependencies']).split(',')
                dependencies = [dep.strip() for dep in deps if dep.strip()]
            
            self.system_dependencies[system_id] = {
                'dependencies': dependencies,
                'type': system.get('type', 'Unknown'),
                'criticality': system.get('criticality', 'Medium'),
                'reliability_score': system.get('reliability_score', 0.9)
            }
        
        # Add ticket-based dependencies (systems frequently affected together)
        system_cooccurrence = self._calculate_system_cooccurrence(tickets_df)
        
        # Add high co-occurrence as implicit dependencies
        threshold = 0.3
        for (sys1, sys2), cooccurrence in system_cooccurrence.items():
            if cooccurrence > threshold:
                if sys1 in self.system_dependencies:
                    if sys2 not in self.system_dependencies[sys1]['dependencies']:
                        self.system_dependencies[sys1]['dependencies'].append(sys2)
    
    def _calculate_system_cooccurrence(self, tickets_df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        """Calculate how often systems are affected together"""
        
        # Group tickets by time windows (e.g., 4-hour windows)
        tickets_df['created_at'] = pd.to_datetime(tickets_df['created_at'])
        tickets_df['time_window'] = tickets_df['created_at'].dt.floor('4H')
        
        # Find systems affected in same time windows
        system_windows = tickets_df.groupby('time_window')['affected_system'].apply(list)
        
        cooccurrence = {}
        
        for systems in system_windows:
            systems = [s for s in systems if pd.notnull(s)]
            
            # Calculate pairwise cooccurrence
            for i, sys1 in enumerate(systems):
                for sys2 in systems[i+1:]:
                    if sys1 != sys2:
                        pair = tuple(sorted([sys1, sys2]))
                        cooccurrence[pair] = cooccurrence.get(pair, 0) + 1
        
        # Normalize by total occurrences
        total_windows = len(system_windows)
        for pair in cooccurrence:
            cooccurrence[pair] /= total_windows
        
        return cooccurrence
    
    def predict_cascade_probability(self, failed_system: str, max_depth: int = 3) -> Dict[str, Any]:
        """Predict probability of cascade failure starting from a system"""
        
        if failed_system not in self.system_dependencies:
            return {'error': f'System {failed_system} not found'}
        
        cascade_risk = {}
        visited = set()
        
        def calculate_cascade_recursive(system_id: str, depth: int, probability: float):
            if depth > max_depth or system_id in visited:
                return
            
            visited.add(system_id)
            
            if system_id not in self.system_dependencies:
                return
            
            system_info = self.system_dependencies[system_id]
            
            # Base failure probability based on system reliability
            base_failure_prob = 1 - system_info['reliability_score']
            
            # Adjust probability based on criticality
            criticality_multiplier = {
                'Critical': 1.5,
                'High': 1.2,
                'Medium': 1.0,
                'Low': 0.8
            }.get(system_info['criticality'], 1.0)
            
            # Current failure probability
            current_prob = min(probability * base_failure_prob * criticality_multiplier, 1.0)
            
            cascade_risk[system_id] = {
                'probability': current_prob,
                'depth': depth,
                'criticality': system_info['criticality'],
                'type': system_info['type']
            }
            
            # Propagate to dependent systems
            for dependent in system_info['dependencies']:
                calculate_cascade_recursive(dependent, depth + 1, current_prob)
        
        # Start cascade prediction
        calculate_cascade_recursive(failed_system, 0, 1.0)
        
        # Calculate overall cascade metrics
        total_systems_at_risk = len(cascade_risk)
        high_risk_systems = sum(1 for risk in cascade_risk.values() if risk['probability'] > 0.5)
        critical_systems_at_risk = sum(1 for risk in cascade_risk.values() 
                                     if risk['criticality'] == 'Critical' and risk['probability'] > 0.3)
        
        return {
            'failed_system': failed_system,
            'cascade_risk': cascade_risk,
            'total_systems_at_risk': total_systems_at_risk,
            'high_risk_systems': high_risk_systems,
            'critical_systems_at_risk': critical_systems_at_risk,
            'max_cascade_probability': max(risk['probability'] for risk in cascade_risk.values()) if cascade_risk else 0
        }

def run_comprehensive_anomaly_detection(tickets_df: pd.DataFrame,
                                       systems_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Run comprehensive anomaly detection analysis"""
    
    logger.info("Starting comprehensive anomaly detection...")
    
    results = {}
    
    # Time series anomaly detection
    ts_detector = TimeSeriesAnomalyDetector()
    time_series_data = ts_detector.prepare_time_series_data(tickets_df)
    volume_anomalies = ts_detector.detect_volume_anomalies(time_series_data)
    results['volume_anomalies'] = volume_anomalies
    
    # Ticket-level anomaly detection
    ticket_detector = TicketAnomalyDetector()
    tickets_processed = ticket_detector.prepare_ticket_features(tickets_df)
    
    resolution_anomalies = ticket_detector.detect_resolution_time_anomalies(tickets_processed)
    behavior_anomalies = ticket_detector.detect_behavior_anomalies(tickets_processed)
    
    results['resolution_time_anomalies'] = resolution_anomalies
    results['behavior_anomalies'] = behavior_anomalies
    
    # Autoencoder anomaly detection (if TensorFlow available)
    if TF_AVAILABLE:
        autoencoder = AutoencoderAnomalyDetector()
        
        # Prepare features for autoencoder
        feature_cols = [col for col in tickets_processed.columns 
                       if tickets_processed[col].dtype in ['int64', 'float64']]
        
        if len(feature_cols) > 10:  # Need sufficient features
            X = tickets_processed[feature_cols].fillna(0).values
            autoencoder.fit(X)
            ae_anomalies, ae_scores = autoencoder.predict(X)
            
            results['autoencoder_anomalies'] = {
                'total_anomalies': ae_anomalies.sum(),
                'anomaly_rate': ae_anomalies.mean(),
                'avg_reconstruction_error': ae_scores.mean()
            }
    
    # Cascade prediction (if systems data available)
    if systems_df is not None:
        cascade_predictor = CascadePredictor()
        cascade_predictor.build_dependency_graph(systems_df, tickets_df)
        
        # Test cascade prediction for critical systems
        critical_systems = systems_df[systems_df['criticality'] == 'Critical']['system_id'].tolist()
        
        cascade_predictions = {}
        for system in critical_systems[:5]:  # Test top 5 critical systems
            prediction = cascade_predictor.predict_cascade_probability(system)
            if 'error' not in prediction:
                cascade_predictions[system] = prediction
        
        results['cascade_predictions'] = cascade_predictions
    
    return results

def main():
    """Main function to run anomaly detection"""
    
    # Load data
    data_path = Path(__file__).parent.parent.parent / "data"
    
    tickets_path = data_path / "synthetic_tickets.csv"
    if not tickets_path.exists():
        logger.error(f"Tickets data not found: {tickets_path}")
        return
    
    logger.info("Loading data...")
    tickets_df = pd.read_csv(tickets_path)
    
    # Load systems data if available
    systems_df = None
    systems_path = data_path / "system_inventory.csv"
    if systems_path.exists():
        systems_df = pd.read_csv(systems_path)
    
    # Run comprehensive anomaly detection
    results = run_comprehensive_anomaly_detection(tickets_df, systems_df)
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("ANOMALY DETECTION RESULTS")
    logger.info("="*50)
    
    # Volume anomalies
    if 'volume_anomalies' in results:
        vol_anom = results['volume_anomalies']
        logger.info(f"\nTicket Volume Anomalies:")
        logger.info(f"  Total anomalies: {vol_anom['total_anomalies']}")
        logger.info(f"  Anomaly rate: {vol_anom['anomaly_rate']:.3f}")
    
    # Resolution time anomalies
    if 'resolution_time_anomalies' in results:
        res_anom = results['resolution_time_anomalies']
        logger.info(f"\nResolution Time Anomalies:")
        logger.info(f"  Total anomalies: {res_anom['total_anomalies']}")
        logger.info(f"  Anomaly rate: {res_anom['anomaly_rate']:.3f}")
        logger.info(f"  Avg resolution time (anomalies): {res_anom['avg_resolution_time_anomalies']:.2f}h")
        logger.info(f"  Avg resolution time (normal): {res_anom['avg_resolution_time_normal']:.2f}h")
    
    # Behavior anomalies
    if 'behavior_anomalies' in results:
        beh_anom = results['behavior_anomalies']
        logger.info(f"\nUser Behavior Anomalies:")
        logger.info(f"  Anomalous users: {beh_anom['total_anomalous_users']}")
        logger.info(f"  User anomaly rate: {beh_anom['anomaly_rate']:.3f}")
    
    # Autoencoder anomalies
    if 'autoencoder_anomalies' in results:
        ae_anom = results['autoencoder_anomalies']
        logger.info(f"\nAutoencoder Anomalies:")
        logger.info(f"  Total anomalies: {ae_anom['total_anomalies']}")
        logger.info(f"  Anomaly rate: {ae_anom['anomaly_rate']:.3f}")
        logger.info(f"  Avg reconstruction error: {ae_anom['avg_reconstruction_error']:.4f}")
    
    # Cascade predictions
    if 'cascade_predictions' in results:
        cascade_pred = results['cascade_predictions']
        logger.info(f"\nCascade Failure Predictions:")
        for system, prediction in cascade_pred.items():
            logger.info(f"  {system}:")
            logger.info(f"    Systems at risk: {prediction['total_systems_at_risk']}")
            logger.info(f"    High risk systems: {prediction['high_risk_systems']}")
            logger.info(f"    Max cascade probability: {prediction['max_cascade_probability']:.3f}")
    
    # Save results
    output_path = data_path / "anomaly_detection_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {output_path}")
    logger.info("Anomaly detection completed successfully!")

if __name__ == "__main__":
    main()