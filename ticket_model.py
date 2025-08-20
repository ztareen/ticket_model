import warnings
import datetime
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

today = date.today()
EVENT_DATA_PATH = f"C:/Users/zarak/OneDrive/Documents/GitHub/ticket_model/event_data_{today.year}.{today.month:02d}.{today.day:02d}.csv"

class TicketTimingOptimizer:
    def __init__(self):
        self.model_classifier = None
        self.model_regressor = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.optimal_timing_patterns = None
        self.price_stats = None
    
    def load_and_prepare_data(self, seat_data_path, event_data_path, target_game, target_date):
        """Load and prepare data for training"""
        try:
            # Load seat data
            df_seats = pd.read_csv(seat_data_path)

            # --- Robustly ensure a 'date' column exists in seat data ---
            seat_date_col = None
            for col in df_seats.columns:
                if col.lower() == 'date':
                    seat_date_col = col
                    break
            if seat_date_col is None:
                # Try to find an alternative date column
                for col in df_seats.columns:
                    if col.lower() in ['event_date', 'start_date']:
                        seat_date_col = col
                        break
            if seat_date_col is not None and seat_date_col != 'date':
                df_seats['date'] = df_seats[seat_date_col]
            elif seat_date_col is None:
                # No date column at all, fill with target_date
                df_seats['date'] = target_date

            # Ensure proper column names (adjust based on your actual CSV structure)
            expected_columns = ['date', 'zone', 'section', 'row', 'seat', 'price', 'quantity']
            if len(df_seats.columns) >= len(expected_columns):
                df_seats.columns = expected_columns[:len(expected_columns)]

            # Load event data if available
            try:
                df_events = pd.read_csv(event_data_path)
                # Robustly find the event date column
                event_date_col = None
                for col in df_events.columns:
                    if col.lower() == 'event_date' or col.lower() == 'start_date':
                        event_date_col = col
                        break
                if event_date_col is not None:
                    df = df_seats.merge(df_events, left_on='date', right_on=event_date_col, how='left')
                else:
                    print(f"No event date column found in event data file: {event_data_path}. Skipping merge.")
                    df = df_seats.copy()
                    # Add default event info
                    df['segment'] = 'Sports'
                    df['genre'] = 'Baseball'
                    df['ticket_status'] = 'Available'
                    df['venue_id'] = 'Unknown'
            except FileNotFoundError:
                print(f"Event data file not found: {event_data_path}")
                df = df_seats.copy()
                # Add default event info
                df['segment'] = 'Sports'
                df['genre'] = 'Baseball'
                df['ticket_status'] = 'Available'
                df['venue_id'] = 'Unknown'

            # Convert date column
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Calculate days until event
            event_date = pd.to_datetime(target_date)
            df['days_until_event'] = (event_date - df['date']).dt.days
            
            # Filter for reasonable time windows
            df = df[(df['days_until_event'] >= 1) & (df['days_until_event'] <= 90)]
            
            if len(df) == 0:
                raise ValueError("No data found within reasonable time window")
            
            # Add time-based features
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['month'] = df['date'].dt.month
            
            # Process the data with feature engineering
            df = self.add_advanced_features(df)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")
    
    def add_advanced_features(self, df):
        """Add advanced features for better prediction"""
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['price', 'quantity', 'days_until_event']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['price', 'days_until_event'])
        
        # Calculate price percentiles for each seat group
        df['price_percentile'] = df.groupby(['zone', 'section', 'row'])['price'].rank(pct=True)
        
        # Price change features (more robust)
        df['price_change'] = df.groupby(['zone', 'section', 'row'])['price'].pct_change()
        df['price_change_7d'] = df.groupby(['zone', 'section', 'row'])['price'].pct_change(periods=7)
        
        # Rolling statistics (shorter windows for more responsive signals)
        df['price_rolling_mean_7d'] = df.groupby(['zone', 'section', 'row'])['price'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        df['price_rolling_min_7d'] = df.groupby(['zone', 'section', 'row'])['price'].transform(
            lambda x: x.rolling(window=7, min_periods=1).min()
        )
        df['price_rolling_max_7d'] = df.groupby(['zone', 'section', 'row'])['price'].transform(
            lambda x: x.rolling(window=7, min_periods=1).max()
        )
        
        # Price position relative to recent range
        df['price_position'] = (df['price'] - df['price_rolling_min_7d']) / (df['price_rolling_max_7d'] - df['price_rolling_min_7d'] + 1e-6)
        
        # Quantity-based features
        if 'quantity' in df.columns:
            df['quantity_change'] = df.groupby(['zone', 'section', 'row'])['quantity'].pct_change()
            df['total_quantity_by_section'] = df.groupby(['section', 'date'])['quantity'].transform('sum')
        else:
            df['quantity_change'] = 0
            df['total_quantity_by_section'] = 100  # Default value
        
        # Time-based price trends
        df['time_trend'] = df['days_until_event'] / df['days_until_event'].max()
        
        # Fill NaN and inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def create_features(self, df):
        """Create feature matrix for training"""
        
        categorical_cols = ['zone', 'section', 'segment', 'genre', 'ticket_status']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    try:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        # If there are unseen categories, assign them to 0
                        df[f'{col}_encoded'] = df[col].astype(str).apply(
                            lambda x: self.label_encoders[col].transform([x])[0] if x in self.label_encoders[col].classes_ else 0
                        )
        
        feature_cols = [
            'days_until_event', 'price', 'quantity', 'day_of_week', 'is_weekend',
            'week_of_year', 'month', 'price_percentile', 'price_change', 'price_change_7d',
            'price_rolling_mean_7d', 'price_position', 'quantity_change',
            'total_quantity_by_section', 'time_trend'
        ]
        
        for col in categorical_cols:
            if f'{col}_encoded' in df.columns:
                feature_cols.append(f'{col}_encoded')
        
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        return df[feature_cols]
    
    def train_model(self, df):
        """Train improved model with baseball-specific logic for games within 30 days"""
        
        X = self.create_features(df)
        
        # IMPROVED: Baseball-specific optimal timing logic for games within 30 days
        # For baseball games in next 30 days, optimal timing is typically:
        # - 7-21 days before for best prices
        # - Bottom 30th percentile of prices (not 25th for more opportunities)
        # - Avoid day-of-game unless desperate
        
        df['is_good_price'] = df.groupby(['zone', 'section', 'row'])['price_percentile'].transform(lambda x: x <= 0.30)
        
        # Baseball-specific timing windows (more lenient for upcoming games)
        df['is_optimal_timing'] = (
            df['is_good_price'] & 
            (df['days_until_event'] >= 3) &  # At least 3 days before (not day-of)
            (df['days_until_event'] <= 25)   # Within 25 days (most baseball games)
        )
        
        # Create target for timing classification
        y_classification = df['is_optimal_timing'].astype(int)
        
        # Store realistic optimal timing patterns
        optimal_windows = df[df['is_optimal_timing'] == 1]
        if len(optimal_windows) > 0:
            self.optimal_timing_patterns = {
                'overall_median': float(optimal_windows['days_until_event'].median()),
                'overall_mean': float(optimal_windows['days_until_event'].mean()),
                'by_section': optimal_windows.groupby('section')['days_until_event'].median().to_dict(),
                'by_zone': optimal_windows.groupby('zone')['days_until_event'].median().to_dict(),
                'by_price_range': {
                    'low': float(optimal_windows[optimal_windows['price'] <= optimal_windows['price'].quantile(0.33)]['days_until_event'].median()),
                    'mid': float(optimal_windows[(optimal_windows['price'] > optimal_windows['price'].quantile(0.33)) & 
                                         (optimal_windows['price'] <= optimal_windows['price'].quantile(0.67))]['days_until_event'].median()),
                    'high': float(optimal_windows[optimal_windows['price'] > optimal_windows['price'].quantile(0.67)]['days_until_event'].median())
                }
            }
        else:
            self.optimal_timing_patterns = {
                'by_section': {},
                'by_zone': {},
                'by_price_range': {}
            }
        
        # Use actual price as regression target
        y_regression = df['price']
        
        # Check if we have enough data for meaningful training
        if len(df) < 10:
            raise ValueError(f"Insufficient data for training. Only {len(df)} samples available. Need at least 10 samples.")
        
        # Check class balance for classification
        n_pos = len(y_classification[y_classification == 1])
        n_neg = len(y_classification[y_classification == 0])
        
        print(f"Training data: {len(df)} samples, {n_pos} optimal timing, {n_neg} non-optimal")
        
        if n_pos < 2 or n_neg < 2:
            # Not enough samples for each class, use simpler approach
            print(f"Warning: Insufficient class balance (positive: {n_pos}, negative: {n_neg}). Using fallback approach.")
            # Try to fit DummyClassifier with both classes if possible
            X_scaled = self.scaler.fit_transform(X)
            unique_classes = np.unique(y_classification)
            if len(unique_classes) == 1:
                # Only one class present, manually add a sample of the other class for fitting
                y_aug = np.append(y_classification, 1 - unique_classes[0])
                X_aug = np.vstack([X_scaled, X_scaled[0]])
                self.model_classifier = DummyClassifier(strategy='stratified', random_state=42)
                self.model_classifier.fit(X_aug, y_aug)
            else:
                self.model_classifier = DummyClassifier(strategy='stratified', random_state=42)
                self.model_classifier.fit(X_scaled, y_classification)
            self.model_regressor = DummyRegressor(strategy='median')
            self.model_regressor.fit(X_scaled, y_regression)
            # Store feature importance (dummy values)
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance_class': [1.0/len(X.columns)] * len(X.columns),
                'importance_reg': [1.0/len(X.columns)] * len(X.columns)
            }).sort_values('importance_class', ascending=False)
            return X_scaled, X_scaled, y_classification, y_regression, y_classification, y_regression, df
        
        # Split data with better error handling
        try:
            X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
                X, y_classification, y_regression, test_size=0.2, random_state=42, stratify=y_classification
            )
        except ValueError as e:
            # If stratification fails, use random split
            print(f"Warning: Stratification failed ({str(e)}). Using random split.")
            X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
                X, y_classification, y_regression, test_size=0.2, random_state=42
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classification model with balanced weights
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        self.model_classifier = xgb.XGBClassifier(
            n_estimators=100,  # Reduced for faster training
            max_depth=4,       # Reduced to prevent overfitting
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
        self.model_classifier.fit(X_train_scaled, y_class_train)
        
        # Train regression model
        self.model_regressor = xgb.XGBRegressor(
            n_estimators=100,  # Reduced for faster training
            max_depth=4,       # Reduced to prevent overfitting
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.model_regressor.fit(X_train_scaled, y_reg_train)
        
        # Evaluate models
        class_pred = self.model_classifier.predict(X_test_scaled)
        class_proba = self.model_classifier.predict_proba(X_test_scaled)
        reg_pred = self.model_regressor.predict(X_test_scaled)
        
        print("=== MODEL PERFORMANCE ===")
        print(f"Classification Accuracy: {np.mean(class_pred == y_class_test):.3f}")
        if class_proba.shape[1] > 1:
            print(f"Average Optimal Probability: {np.mean(class_proba[:, 1]):.3f}")
        print(f"Regression R²: {r2_score(y_reg_test, reg_pred):.3f}")
        print(f"Regression MAE: {mean_absolute_error(y_reg_test, reg_pred):.3f}")
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance_class': self.model_classifier.feature_importances_,
            'importance_reg': self.model_regressor.feature_importances_
        }).sort_values('importance_class', ascending=False)
        
        return X_train_scaled, X_test_scaled, y_class_test, y_reg_test, class_pred, reg_pred, df
    
    def predict_optimal_timing(self, current_features):
        """FIXED: Predict optimal timing with dynamic probability calculation"""
        if self.model_classifier is None or self.model_regressor is None:
            raise ValueError("Models not trained yet!")

        current_features_scaled = self.scaler.transform(current_features.reshape(1, -1))

        # Get probabilities and predictions with robust error handling
        try:
            # Try to get probability predictions
            proba = self.model_classifier.predict_proba(current_features_scaled)
            
            # Handle different cases based on model output
            if proba.shape[1] == 2:
                # Standard binary classification case
                raw_prob = float(proba[0][1])  # Probability of class 1 (optimal)
                print(f"Model raw probability: {raw_prob}")
            elif proba.shape[1] == 1:
                # Only one class in training data
                unique_classes = self.model_classifier.classes_
                if len(unique_classes) == 1:
                    if unique_classes[0] == 1:
                        # Only optimal class seen during training
                        raw_prob = 0.75  # High probability since only optimal class exists
                    else:
                        # Only non-optimal class seen during training  
                        raw_prob = 0.25  # Low probability since only non-optimal exists
                else:
                    raw_prob = float(proba[0][0])
                print(f"Single class model probability: {raw_prob}")
            else:
                # Unexpected case
                raw_prob = 0.5
                print(f"Unexpected case, using default: {raw_prob}")
                
        except Exception as e:
            print(f"Warning: Probability prediction failed: {e}")
            # Fallback: Calculate probability based on features
            raw_prob = self._calculate_fallback_probability(current_features)
            print(f"Fallback probability: {raw_prob}")
        
        # Get feature-based probability for comparison/adjustment
        feature_prob = self._calculate_feature_based_probability(current_features)
        print(f"Feature-based probability: {feature_prob}")
        
        # Combine model and feature probabilities intelligently
        if hasattr(self.model_classifier, '__class__') and 'Dummy' in str(self.model_classifier.__class__):
            # If using dummy classifier, rely more on feature-based calculation
            optimal_prob = 0.3 * raw_prob + 0.7 * feature_prob
        else:
            # If using real model, balance both approaches
            optimal_prob = 0.6 * raw_prob + 0.4 * feature_prob
        
        print(f"Combined probability before bounds: {optimal_prob}")
        
        # Apply reasonable bounds (but don't force to minimum)
        if optimal_prob < 0.05:
            optimal_prob = 0.08  # Very low but not zero
        elif optimal_prob > 0.95:
            optimal_prob = 0.92  # Very high but not certain
        
        print(f"Final probability: {optimal_prob}")
        
        # Predict price
        try:
            predicted_price = self.model_regressor.predict(current_features_scaled)[0]
            if pd.isna(predicted_price) or predicted_price <= 0:
                predicted_price = 100.0  # Fallback price
        except Exception as e:
            print(f"Warning: Price prediction failed: {e}")
            predicted_price = 100.0  # Fallback price

        return float(optimal_prob), float(predicted_price)
    
    def _calculate_fallback_probability(self, current_features):
        """Calculate probability based on feature analysis when model prediction fails"""
        try:
            # Extract key features if available
            days_until_event = None
            price_percentile = None
            price_position = None
            
            # Try to find features using the scaler's feature names
            if hasattr(self, 'scaler') and hasattr(self.scaler, 'feature_names_in_'):
                feature_names = self.scaler.feature_names_in_
            else:
                # If no feature names, try to infer from common feature order
                feature_names = ['days_until_event', 'price', 'quantity', 'day_of_week', 'is_weekend',
                               'week_of_year', 'month', 'price_percentile', 'price_change', 'price_change_7d',
                               'price_rolling_mean_7d', 'price_position', 'quantity_change',
                               'total_quantity_by_section', 'time_trend']
                feature_names = feature_names[:len(current_features)]
            
            # Extract key feature values
            for i, feature_name in enumerate(feature_names):
                if feature_name == 'days_until_event' and i < len(current_features):
                    days_until_event = current_features[i]
                elif feature_name == 'price_percentile' and i < len(current_features):
                    price_percentile = current_features[i]
                elif feature_name == 'price_position' and i < len(current_features):
                    price_position = current_features[i]
            
            # Calculate probability based on baseball-specific logic
            base_prob = 0.5
            
            # Adjust for days until event (optimal window: 7-21 days for baseball)
            if days_until_event is not None:
                if 7 <= days_until_event <= 21:
                    base_prob += 0.2  # In optimal window
                elif 3 <= days_until_event < 7 or 21 < days_until_event <= 25:
                    base_prob += 0.1  # Close to optimal
                elif days_until_event < 3:
                    base_prob -= 0.1  # Too close to event
                else:
                    base_prob -= 0.15  # Too far from event
            
            # Adjust for price percentile (lower is better)
            if price_percentile is not None:
                if price_percentile <= 0.3:
                    base_prob += 0.15  # Good price
                elif price_percentile <= 0.5:
                    base_prob += 0.05  # Decent price
                else:
                    base_prob -= 0.1   # Higher price
            
            # Adjust for price position (lower is better)
            if price_position is not None:
                if price_position <= 0.3:
                    base_prob += 0.1   # Low in recent range
                elif price_position >= 0.7:
                    base_prob -= 0.1   # High in recent range
            
            # Ensure bounds
            return max(0.1, min(0.9, base_prob))
            
        except Exception as e:
            print(f"Warning: Fallback probability calculation failed: {e}")
            return 0.5
    
    def _calculate_feature_based_probability(self, current_features):
        """Calculate probability based on feature analysis"""
        try:
            # Similar to fallback probability but with different logic
            days_until_event = None
            price_percentile = None
            
            # Extract features
            if hasattr(self, 'scaler') and hasattr(self.scaler, 'feature_names_in_'):
                feature_names = self.scaler.feature_names_in_
            else:
                feature_names = ['days_until_event', 'price', 'quantity', 'day_of_week', 'is_weekend',
                               'week_of_year', 'month', 'price_percentile', 'price_change', 'price_change_7d',
                               'price_rolling_mean_7d', 'price_position', 'quantity_change',
                               'total_quantity_by_section', 'time_trend']
                feature_names = feature_names[:len(current_features)]
            
            for i, feature_name in enumerate(feature_names):
                if feature_name == 'days_until_event' and i < len(current_features):
                    days_until_event = current_features[i]
                elif feature_name == 'price_percentile' and i < len(current_features):
                    price_percentile = current_features[i]
            
            # Use optimal timing patterns if available
            if hasattr(self, 'optimal_timing_patterns') and self.optimal_timing_patterns:
                optimal_days = self.optimal_timing_patterns.get('overall_median', 14)
                
                if days_until_event is not None:
                    distance = abs(days_until_event - optimal_days)
                    if distance <= 3:
                        return 0.8
                    elif distance <= 7:
                        return 0.6
                    else:
                        return 0.3
            
            return 0.5
            
        except Exception as e:
            print(f"Warning: Feature-based probability calculation failed: {e}")
            return 0.5
    
    def predict_future_optimal_timing(self, current_days_until_event, section=None, zone=None, current_price=None):
        """Predict future optimal timing windows based on learned patterns"""
        
        if not hasattr(self, 'optimal_timing_patterns') or not self.optimal_timing_patterns:
            # Baseball-specific fallback for games within 30 days
            if current_days_until_event <= 30:
                return max(3, min(14, current_days_until_event * 0.7))
            return max(7, min(21, current_days_until_event * 0.5))
        
        # Start with overall median as baseline
        base_optimal_days = self.optimal_timing_patterns.get('overall_median', 14)
        
        # Adjust based on section
        if section and section in self.optimal_timing_patterns.get('by_section', {}):
            section_optimal = self.optimal_timing_patterns['by_section'][section]
            if not pd.isna(section_optimal):
                base_optimal_days = (base_optimal_days + section_optimal) / 2
        
        # Adjust based on zone
        if zone and zone in self.optimal_timing_patterns.get('by_zone', {}):
            zone_optimal = self.optimal_timing_patterns['by_zone'][zone]
            if not pd.isna(zone_optimal):
                base_optimal_days = (base_optimal_days + zone_optimal) / 2
        
        # Baseball-specific bounds for games within 30 days
        if current_days_until_event <= 30:
            base_optimal_days = max(3, min(25, base_optimal_days))
        else:
            base_optimal_days = max(7, min(60, base_optimal_days))
        
        # If we're past the optimal window, suggest buying soon
        if base_optimal_days > current_days_until_event:
            if current_days_until_event <= 7:
                return max(1, current_days_until_event)
            else:
                return max(3, min(7, current_days_until_event // 2))
        
        return max(1, base_optimal_days)
    
    def analyze_timing_patterns(self, df):
        """Analyze patterns in optimal timing with improved metrics"""
        
        # Focus on realistic optimal windows
        optimal_data = df[df.get('is_optimal_timing', False) == True]
        
        if len(optimal_data) == 0:
            # Fallback analysis using price percentiles
            good_prices = df[df['price_percentile'] <= 0.3]
            if len(good_prices) > 0:
                # Baseball-specific bounds for games within 30 days
                bounded_good_prices = good_prices[
                    (good_prices['days_until_event'] >= 3) & 
                    (good_prices['days_until_event'] <= 25)
                ]
                if len(bounded_good_prices) > 0:
                    timing_analysis = {
                        'overall_avg_days': float(bounded_good_prices['days_until_event'].mean()),
                        'overall_median_days': float(bounded_good_prices['days_until_event'].median()),
                        'by_zone': bounded_good_prices.groupby('zone')['days_until_event'].median().to_dict(),
                        'by_section': bounded_good_prices.groupby('section')['days_until_event'].median().to_dict(),
                        'by_month': bounded_good_prices.groupby('month')['days_until_event'].median().to_dict(),
                        'price_range_analysis': {
                            'min_price': float(df['price'].min()),
                            'max_price': float(df['price'].max()),
                            '25th_percentile': float(df['price'].quantile(0.25)),
                            '75th_percentile': float(df['price'].quantile(0.75))
                        }
                    }
                else:
                    timing_analysis = {
                        'by_zone': {},
                        'by_section': {},
                        'by_month': {},
                        'price_range_analysis': {
                            'min_price': float(df['price'].min()),
                            'max_price': float(df['price'].max()),
                            '25th_percentile': float(df['price'].quantile(0.25)),
                            '75th_percentile': float(df['price'].quantile(0.75))
                        }
                    }
            else:
                timing_analysis = {
                    'by_zone': {},
                    'by_section': {},
                    'by_month': {},
                    'price_range_analysis': {
                        'min_price': float(df['price'].min()),
                        'max_price': float(df['price'].max()),
                        '25th_percentile': float(df['price'].quantile(0.25)),
                        '75th_percentile': float(df['price'].quantile(0.75))
                    }
                }
        else:
            # Baseball-specific bounds
            bounded_optimal_data = optimal_data[
                (optimal_data['days_until_event'] >= 3) & 
                (optimal_data['days_until_event'] <= 25)
            ]
            if len(bounded_optimal_data) == 0:
                timing_analysis = {
                    'by_zone': {},
                    'by_section': {},
                    'by_month': {},
                    'price_range_analysis': {
                        'min_price': float(df['price'].min()),
                        'max_price': float(df['price'].max()),
                        '25th_percentile': float(df['price'].quantile(0.25)),
                        '75th_percentile': float(df['price'].quantile(0.75))
                    }
                }
            else:
                timing_analysis = {
                    'overall_avg_days': float(bounded_optimal_data['days_until_event'].mean()),
                    'overall_median_days': float(bounded_optimal_data['days_until_event'].median()),
                    'by_zone': bounded_optimal_data.groupby('zone')['days_until_event'].median().to_dict(),
                    'by_section': bounded_optimal_data.groupby('section')['days_until_event'].median().to_dict(),
                    'by_month': bounded_optimal_data.groupby('month')['days_until_event'].median().to_dict(),
                    'price_range_analysis': {
                        'min_price': float(df['price'].min()),
                        'max_price': float(df['price'].max()),
                        '25th_percentile': float(df['price'].quantile(0.25)),
                        '75th_percentile': float(df['price'].quantile(0.75))
                    }
                }
        return timing_analysis
    
    def visualize_results(self, df, df_with_optimal=None):
        """Create visualizations of the analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Price vs Days scatter
        axes[0, 0].scatter(df['days_until_event'], df['price'], alpha=0.6, s=10)
        axes[0, 0].set_xlabel('Days Until Event')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].set_title('Price vs Days Until Event')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Optimal timing distribution
        if 'is_optimal_timing' in df.columns:
            optimal_days = df[df['is_optimal_timing'] == 1]['days_until_event']
            if len(optimal_days) > 0:
                axes[0, 1].hist(optimal_days, bins=20, alpha=0.7, edgecolor='black')
                axes[0, 1].set_xlabel('Optimal Days Before Event')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Distribution of Optimal Purchase Timing')
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'No optimal timing data available', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Feature importance
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[1, 0].barh(range(len(top_features)), top_features['importance_class'])
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features['feature'])
            axes[1, 0].set_xlabel('Feature Importance')
            axes[1, 0].set_title('Top 10 Features for Timing Prediction')
        
        # Price percentiles over time
        if 'price_percentile' in df.columns:
            axes[1, 1].scatter(df['days_until_event'], df['price_percentile'], alpha=0.6, s=10)
            axes[1, 1].axhline(y=0.30, color='red', linestyle='--', label='Good Price Threshold')
            axes[1, 1].set_xlabel('Days Until Event')
            axes[1, 1].set_ylabel('Price Percentile')
            axes[1, 1].set_title('Price Percentiles Over Time')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


def main():
    optimizer = TicketTimingOptimizer()
    return optimizer, None, None


def run_ticket_model(seat_data_path, event_data_path, target_game, target_date):
    """FIXED main function with robust probability calculation for baseball games"""
    
    try:
        optimizer = TicketTimingOptimizer()
        df = optimizer.load_and_prepare_data(seat_data_path, event_data_path, target_game, target_date)
        # Remove price outliers using IQR method
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(1, Q1 - 1.5 * IQR)
        upper_bound = min(10000, Q3 + 1.5 * IQR)
        df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
        X_train, X_test, y_class_test, y_reg_test, class_pred, reg_pred, df_processed = optimizer.train_model(df)
        timing_analysis = optimizer.analyze_timing_patterns(df_processed)

        # Store price statistics for comparison (after outlier removal)
        optimizer.price_stats = {
            'min': float(df['price'].min()),
            'max': float(df['price'].max()),
            'median': float(df['price'].median()),
            '25th': float(df['price'].quantile(0.25)),
            '75th': float(df['price'].quantile(0.75))
        }

        # Calculate actual current days until event
        try:
            event_date = datetime.datetime.strptime(target_date, '%Y-%m-%d')
            current_date = datetime.datetime.now()
            current_days = (event_date - current_date).days
            current_days = max(1, min(365, current_days))
        except:
            current_days = 15  # Fallback to 15 days for baseball
        
        print(f"Current days until event: {current_days}")
        
        # Get consistent timing recommendations
        # Make average_best_days deterministic and based on available data
        if 'overall_median_days' in timing_analysis and not pd.isna(timing_analysis['overall_median_days']):
            average_best_days = float(timing_analysis['overall_median_days'])
        elif 'overall_avg_days' in timing_analysis and not pd.isna(timing_analysis['overall_avg_days']):
            average_best_days = float(timing_analysis['overall_avg_days'])
        else:
            # Fallback: use 14 as a typical value for baseball
            average_best_days = 14.0
        average_best_days = max(3, min(25, average_best_days))
        print(f"Average best days: {average_best_days}")

        # FIXED: Create more representative current features for probability prediction
        # Use median/representative values from the actual data rather than just the most recent
        feature_cols = optimizer.create_features(df_processed).columns
        current_features = np.zeros(len(feature_cols))
        
        # Set features based on current situation and data patterns
        for i, col in enumerate(feature_cols):
            if col == 'days_until_event':
                current_features[i] = current_days
            elif col == 'price':
                # Use median price as representative current price
                current_features[i] = float(df['price'].median())
            elif col == 'quantity':
                if 'quantity' in df.columns:
                    current_features[i] = float(df['quantity'].median())
                else:
                    current_features[i] = 100  # Default value
            elif col == 'day_of_week':
                current_features[i] = datetime.datetime.now().weekday()
            elif col == 'is_weekend':
                current_features[i] = 1 if datetime.datetime.now().weekday() >= 5 else 0
            elif col == 'month':
                current_features[i] = datetime.datetime.now().month
            elif col == 'week_of_year':
                current_features[i] = datetime.datetime.now().isocalendar()[1]
            elif col == 'time_trend':
                current_features[i] = current_days / 365
            elif col == 'price_percentile':
                # Calculate where median price falls in overall distribution
                median_price = df['price'].median()
                price_rank = (median_price - df['price'].min()) / (df['price'].max() - df['price'].min() + 1e-6)
                current_features[i] = price_rank
            elif col == 'price_position':
                # Use median price position
                current_features[i] = 0.5  # Middle of range
            elif col == 'price_rolling_mean_7d':
                current_features[i] = float(df['price'].median())
            elif 'encoded' in col:
                # Use mode (most common) for categorical features
                base_col = col.replace('_encoded', '')
                if base_col in df.columns:
                    mode_val = df[base_col].mode()
                    if len(mode_val) > 0:
                        mode_str = str(mode_val.iloc[0])
                        if base_col in optimizer.label_encoders:
                            try:
                                if mode_str in optimizer.label_encoders[base_col].classes_:
                                    current_features[i] = optimizer.label_encoders[base_col].transform([mode_str])[0]
                                else:
                                    current_features[i] = 0
                            except:
                                current_features[i] = 0
            else:
                # For other features, use median or zero
                if col in df.columns:
                    col_median = df[col].median()
                    if not pd.isna(col_median):
                        current_features[i] = float(col_median)
                    else:
                        current_features[i] = 0
                else:
                    current_features[i] = 0

        print(f"Feature vector created with {len(current_features)} features")
        
        # Get probability prediction with the improved method
        optimal_prob, predicted_price = optimizer.predict_optimal_timing(current_features)

        print(f"Raw optimal probability: {optimal_prob}")
        print(f"Raw predicted price: {predicted_price}")

        # Ensure we have valid values with better bounds
        if pd.isna(optimal_prob) or optimal_prob < 0.1 or optimal_prob > 1.0:
            # Fallback calculation based on timing
            days_until_optimal = current_days - average_best_days
            if abs(days_until_optimal) <= 3:
                optimal_prob = 0.75  # Close to optimal
            elif abs(days_until_optimal) <= 7:
                optimal_prob = 0.55  # Somewhat close
            elif days_until_optimal < -10:
                optimal_prob = 0.35  # Past optimal window
            else:
                optimal_prob = 0.45  # Default reasonable value
        
        # Ensure probability is in reasonable range
        optimal_prob = max(0.15, min(0.90, float(optimal_prob)))
        
        # Validate predicted price
        predicted_price = float(predicted_price) if not pd.isna(predicted_price) else float(df['price'].median())
        predicted_price = max(float(df['price'].min()), min(predicted_price, float(df['price'].max())))

        # Feature importance (top 3)
        if optimizer.feature_importance is not None:
            top_features = optimizer.feature_importance.head(3)['feature'].tolist()
        else:
            top_features = ['days_until_event', 'price', 'price_percentile']

        # Clean up timing data
        timing_by_zone = {}
        timing_by_section = {}

        for k, v in timing_analysis.get('by_zone', {}).items():
            if not pd.isna(v):
                timing_by_zone[str(k)] = round(float(v), 1)

        for k, v in timing_analysis.get('by_section', {}).items():
            if not pd.isna(v):
                timing_by_section[str(k)] = round(float(v), 1)

        # Price analysis
        price_stats = timing_analysis.get('price_range_analysis', {})
        price_min = float(price_stats.get('min_price', df['price'].min()))
        price_max = float(price_stats.get('max_price', df['price'].max()))
        price_25th = float(price_stats.get('25th_percentile', df['price'].quantile(0.25)))
        price_75th = float(price_stats.get('75th_percentile', df['price'].quantile(0.75)))

        price_15th = float(df['price'].quantile(0.15))
        # Lower bound: 15th percentile, fallback to min if NaN
        buy_lower = price_15th if not pd.isna(price_15th) else price_min
        # Upper bound: min(25th percentile, median price)
        buy_upper = min(price_25th, float(df['price'].median()))
        buy_price_range = [round(buy_lower, 2), round(buy_upper, 2)]
        price_range = [round(price_min, 2), round(price_max, 2)]
        predicted_price_range = [round(price_25th, 2), round(price_75th, 2)]

        # Section-specific analysis
        price_range_by_section = {}
        buy_price_by_section = {}
        predicted_price_by_section = {}
        buy_days_by_section = timing_by_section.copy()

        for section in df['section'].unique():
            section_data = df[df['section'] == section]
            if len(section_data) > 0:
                try:
                    section_min = float(section_data['price'].min())
                    section_max = float(section_data['price'].max())
                    section_15th = float(section_data['price'].quantile(0.15))
                    section_25th = float(section_data['price'].quantile(0.25))
                    section_median = float(section_data['price'].median())

                    # Lower bound: 15th percentile, fallback to min if NaN
                    sec_buy_lower = section_15th if not pd.isna(section_15th) else section_min
                    # Upper bound: min(25th percentile, median)
                    sec_buy_upper = min(section_25th, section_median)

                    if not any(pd.isna([section_min, section_max, section_25th, section_median, sec_buy_lower, sec_buy_upper])):
                        price_range_by_section[str(section)] = [round(section_min, 2), round(section_max, 2)]
                        buy_price_by_section[str(section)] = [round(sec_buy_lower, 2), round(sec_buy_upper, 2)]
                        predicted_price_by_section[str(section)] = round(section_median, 2)
                    else:
                        price_range_by_section[str(section)] = price_range
                        buy_price_by_section[str(section)] = buy_price_range
                        predicted_price_by_section[str(section)] = round(price_25th, 2)
                except Exception as e:
                    print(f"Warning: Section analysis failed for {section}: {e}")
                    price_range_by_section[str(section)] = price_range
                    buy_price_by_section[str(section)] = buy_price_range
                    predicted_price_by_section[str(section)] = round(price_25th, 2)

        # Recommendation logic based on optimal buy days
        days_until_optimal = int(round(current_days - average_best_days))
        
        if days_until_optimal > 1:
            recommendation = f"We do not recommend buying right now. You should buy in {days_until_optimal} days."
            rec_code = 'yellow'
        elif days_until_optimal == 1:
            recommendation = "We do not recommend buying right now. You should buy tomorrow."
            rec_code = 'yellow'
        elif days_until_optimal == 0:
            recommendation = "Now is the optimal time to buy."
            rec_code = 'green'
        else:  # days_until_optimal < 0
            recommendation = "Optimal time has passed. Buy as soon as possible."
            rec_code = 'yellow'

        # Event details
        event_details = {}
        for col in ['venue_id', 'segment', 'genre', 'ticket_status']:
            if col in df.columns:
                event_details[col] = str(df.iloc[0][col])

        # Improved probability context
        prob_percentage = int(optimal_prob * 100)
        prob_context = (
            f"Based on historical data analysis, current market conditions suggest a {prob_percentage}% "
            f"probability that this is a good time to buy tickets. This calculation considers factors like "
            f"current pricing trends, days until the event ({current_days} days), and historical patterns "
            f"for similar baseball games. The optimal buying window is typically around {int(average_best_days)} days before the event."
        )

        # Final validation of predicted price
        if predicted_price < price_25th * 0.7:
            predicted_price = price_25th * 0.8

        return {
            'event': target_game,
            'event_date': target_date,
            'event_details': event_details,
            'optimal_probability': round(optimal_prob, 2),
            'probability_context': prob_context,
            'predicted_price': round(predicted_price, 2),
            'predicted_price_range': predicted_price_range,
            'average_best_days': round(average_best_days, 1),
            'buy_days_overall': round(average_best_days, 1),
            'feature_importance': top_features,
            'timing_analysis': timing_analysis,
            'timing_by_zone': timing_by_zone,
            'timing_by_section': timing_by_section,
            'price_range': price_range,
            'price_range_by_section': price_range_by_section,
            'buy_price_range': buy_price_range,
            'recommendation': recommendation,
            'recommendation_code': rec_code,
            'predicted_price_by_section': predicted_price_by_section,
            'buy_price_by_section': buy_price_by_section,
            'buy_days_by_section': buy_days_by_section
        }
        
    except Exception as e:
        # Return a fallback result if model training fails
        print(f"Warning: Model training failed for {target_game}: {str(e)}")

        # Calculate current days for fallback
        try:
            event_date_dt = datetime.datetime.strptime(target_date, '%Y-%m-%d')
            current_date = datetime.datetime.now()
            current_days = (event_date_dt - current_date).days
            current_days = max(1, min(30, current_days))
        except Exception:
            current_days = 15

        # Try to get basic price information from the data file
        try:
            df_basic = pd.read_csv(seat_data_path)
            if len(df_basic.columns) >= 6:
                prices = pd.to_numeric(df_basic.iloc[:, 5], errors='coerce').dropna()
                if len(prices) > 0:
                    price_min = float(prices.min())
                    price_max = float(prices.max())
                    price_median = float(prices.median())
                    price_25th = float(prices.quantile(0.25))
                else:
                    price_min, price_max, price_median, price_25th = 50.0, 200.0, 100.0, 75.0
            else:
                price_min, price_max, price_median, price_25th = 50.0, 200.0, 100.0, 75.0
        except:
            price_min, price_max, price_median, price_25th = 50.0, 200.0, 100.0, 75.0

        # Fallback optimal days and probability calculation
        average_best_days = 14.0
        days_until_optimal = int(round(current_days - average_best_days))
        
        # More nuanced fallback probability based on timing
        if days_until_optimal > 7:
            optimal_prob = 0.25  # Too early
            recommendation = f"We do not recommend buying right now. You should buy in {days_until_optimal} days."
            rec_code = 'yellow'
        elif days_until_optimal > 1:
            optimal_prob = 0.45  # Getting closer
            recommendation = f"We do not recommend buying right now. You should buy in {days_until_optimal} days."
            rec_code = 'yellow'
        elif days_until_optimal == 1:
            optimal_prob = 0.65  # Close to optimal
            recommendation = "We do not recommend buying right now. You should buy tomorrow."
            rec_code = 'yellow'
        elif days_until_optimal == 0:
            optimal_prob = 0.80  # Optimal time
            recommendation = "Now is the optimal time to buy."
            rec_code = 'green'
        else:  # days_until_optimal < 0
            optimal_prob = max(0.35, 0.80 - abs(days_until_optimal) * 0.05)  # Decreasing as time passes
            recommendation = "Optimal time has passed. Buy as soon as possible."
            rec_code = 'yellow'

        return {
            'event': target_game,
            'event_date': target_date,
            'event_details': {'segment': 'Sports', 'genre': 'Baseball'},
            'optimal_probability': round(optimal_prob, 2),
            'probability_context': f"Limited data available. Using baseball-specific timing patterns for games {current_days} days away. Probability based on distance from optimal {int(average_best_days)}-day window.",
            'predicted_price': round(price_median, 2),
            'predicted_price_range': [round(price_25th, 2), round(price_max, 2)],
            'average_best_days': average_best_days,
            'buy_days_overall': average_best_days,
            'feature_importance': ['days_until_event', 'price', 'quantity'],
            'timing_analysis': {'overall_median_days': average_best_days},
            'timing_by_zone': {},
            'timing_by_section': {},
            'price_range': [round(price_min, 2), round(price_max, 2)],
            'price_range_by_section': {},
            'buy_price_range': [round(price_25th, 2), round(price_25th * 1.1, 2)],
            'recommendation': recommendation,
            'recommendation_code': rec_code,
            'predicted_price_by_section': {},
            'buy_price_by_section': {},
            'buy_days_by_section': {}
        }


if __name__ == "__main__":
    pass