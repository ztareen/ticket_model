
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import math
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import datetime
from datetime import date, timedelta
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
        self.optimal_timing_days = None
        
    def load_and_prepare_data(self, seat_data_path, event_data_path, target_game, target_date):
        """Load seat data and event metadata, then prepare features"""
        
        # Load seat data
        seat_data = []
        try:
            with open(seat_data_path, 'r') as file:
                csvreader = csv.reader(file)
                header = next(csvreader)
                
                if len(header) < 6:
                    raise ValueError(f"CSV file {seat_data_path} has insufficient columns. Expected at least 6 columns, got {len(header)}")
                
                for row in csvreader:
                    if len(row) < 6:
                        continue
                    seat_data.append({
                        'date': row[0],
                        'zone': row[1],
                        'section': row[2],
                        'row': row[3],
                        'quantity': int(row[4]),
                        'price': float(row[5])
                    })
        except FileNotFoundError:
            raise FileNotFoundError(f"Seat data file not found: {seat_data_path}")
        except Exception as e:
            raise Exception(f"Error reading seat data file {seat_data_path}: {str(e)}")
        
        if not seat_data:
            raise ValueError(f"No valid data found in seat data file: {seat_data_path}")
        
        # Load event metadata
        event_metadata = {}
        try:
            with open(event_data_path, 'r', encoding='utf-8') as file:
                csvreader = csv.DictReader(file)
                for row in csvreader:
                    if (row["start_date"] == target_date or 
                        target_date in row.get("start_date", "") or
                        target_game in row.get("event_name", "")):
                        event_metadata = row
                        break
        except FileNotFoundError:
            print(f"Warning: Event data file not found: {event_data_path}")
        except Exception as e:
            print(f"Warning: Error reading event data file {event_data_path}: {str(e)}")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(seat_data)
        
        # Parse dates and calculate time features
        try:
            df['date'] = pd.to_datetime(df['date'])
            event_date = pd.to_datetime(target_date)
        except Exception as e:
            raise Exception(f"Error parsing dates. Check that date format is correct in {seat_data_path}: {str(e)}")
        
        # Calculate days until event (key feature for timing)
        df['days_until_event'] = (event_date - df['date']).dt.days
        
        # Filter out unrealistic data (more than 365 days or negative days)
        df = df[(df['days_until_event'] >= 0) & (df['days_until_event'] <= 365)]
        
        # Add time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        
        # Add event metadata features
        if event_metadata:
            df['venue_id'] = event_metadata.get('venue_id', 0)
            df['segment'] = event_metadata.get('segment', 'Unknown')
            df['genre'] = event_metadata.get('genre', 'Unknown')
            df['ticket_status'] = event_metadata.get('ticket_status', 'Unknown')
        else:
            df['venue_id'] = 0
            df['segment'] = 'Sports'  # Default to Sports for baseball
            df['genre'] = 'Baseball'  # Default to Baseball
            df['ticket_status'] = 'OnSale'
        
        # Create price trend features
        df = df.sort_values(['zone', 'section', 'row', 'date'])
        
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
        df['quantity_change'] = df.groupby(['zone', 'section', 'row'])['quantity'].pct_change()
        df['total_quantity_by_section'] = df.groupby(['section', 'date'])['quantity'].transform('sum')
        
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
                'overall_median': optimal_windows['days_until_event'].median(),
                'overall_mean': optimal_windows['days_until_event'].mean(),
                'by_section': optimal_windows.groupby('section')['days_until_event'].median().to_dict(),
                'by_zone': optimal_windows.groupby('zone')['days_until_event'].median().to_dict(),
                'by_price_range': {
                    'low': optimal_windows[optimal_windows['price'] <= optimal_windows['price'].quantile(0.33)]['days_until_event'].median(),
                    'mid': optimal_windows[(optimal_windows['price'] > optimal_windows['price'].quantile(0.33)) & 
                                         (optimal_windows['price'] <= optimal_windows['price'].quantile(0.67))]['days_until_event'].median(),
                    'high': optimal_windows[optimal_windows['price'] > optimal_windows['price'].quantile(0.67)]['days_until_event'].median()
                }
            }
        else:
            # Baseball-specific fallback for games within 30 days
            self.optimal_timing_patterns = {
                'overall_median': 14,  # 2 weeks before is typically good for baseball
                'overall_mean': 16,
                'by_section': {},
                'by_zone': {},
                'by_price_range': {'low': 10, 'mid': 14, 'high': 18}
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
            
            # Use dummy models for both classification and regression
            from sklearn.dummy import DummyClassifier, DummyRegressor
            
            # Use the most frequent class instead of constant to avoid errors
            self.model_classifier = DummyClassifier(strategy='most_frequent')
            self.model_regressor = DummyRegressor(strategy='median')
            
            # Fit models on all data since we can't split
            X_scaled = self.scaler.fit_transform(X)
            self.model_classifier.fit(X_scaled, y_classification)
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
        """Predict optimal timing for ticket purchase - FIXED to return valid probability"""
        if self.model_classifier is None or self.model_regressor is None:
            raise ValueError("Models not trained yet!")

        current_features_scaled = self.scaler.transform(current_features.reshape(1, -1))

        # Get probabilities and predictions - FIXED
        try:
            proba = self.model_classifier.predict_proba(current_features_scaled)
            if proba.shape[1] == 1:
                # Only one class present - check which class it is
                unique_classes = self.model_classifier.classes_
                if len(unique_classes) == 1 and unique_classes[0] == 1:
                    optimal_prob = 0.8  # If only optimal class, high probability
                else:
                    optimal_prob = 0.2  # If only non-optimal class, low probability
            else:
                optimal_prob = proba[0][1]  # Probability of class 1 (optimal)
                
        except Exception as e:
            print(f"Warning: Probability calculation failed: {e}")
            optimal_prob = 0.5  # Fallback probability
        
        # Ensure probability is valid and not NaN
        if optimal_prob is None or pd.isna(optimal_prob) or optimal_prob < 0 or optimal_prob > 1:
            optimal_prob = 0.5  # Fallback to reasonable default
        
        # Additional validation to ensure probability is never N/A
        if optimal_prob is None or pd.isna(optimal_prob):
            optimal_prob = 0.5
        
        # Predict price
        try:
            predicted_price = self.model_regressor.predict(current_features_scaled)[0]
            if pd.isna(predicted_price) or predicted_price <= 0:
                predicted_price = 100.0  # Fallback price
        except Exception as e:
            print(f"Warning: Price prediction failed: {e}")
            predicted_price = 100.0  # Fallback price

        return float(optimal_prob), float(predicted_price)
    
    def predict_future_optimal_timing(self, current_days_until_event, section=None, zone=None, current_price=None):
        """Predict future optimal timing windows based on learned patterns"""
        
        if not hasattr(self, 'optimal_timing_patterns'):
            # Baseball-specific fallback for games within 30 days
            if current_days_until_event <= 30:
                return max(3, min(14, current_days_until_event * 0.7))
            return max(7, min(21, current_days_until_event * 0.5))
        
        # Start with overall median as baseline
        base_optimal_days = self.optimal_timing_patterns.get('overall_median', 14)
        
        # Adjust based on section
        if section and section in self.optimal_timing_patterns['by_section']:
            section_optimal = self.optimal_timing_patterns['by_section'][section]
            if not pd.isna(section_optimal):
                base_optimal_days = (base_optimal_days + section_optimal) / 2
        
        # Adjust based on zone
        if zone and zone in self.optimal_timing_patterns['by_zone']:
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
                    # Baseball-specific fallback
                    timing_analysis = {
                        'overall_avg_days': 16.0,
                        'overall_median_days': 14.0,
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
                # Ultimate fallback for baseball
                timing_analysis = {
                    'overall_avg_days': 16.0,
                    'overall_median_days': 14.0,
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
                # Baseball-specific fallback
                timing_analysis = {
                    'overall_avg_days': 16.0,
                    'overall_median_days': 14.0,
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

if __name__ == "__main__":
    pass

def run_ticket_model(seat_data_path, event_data_path, target_game, target_date):
    """FIXED main function with consistent logic for baseball games within 30 days"""
    
    try:
        optimizer = TicketTimingOptimizer()
        df = optimizer.load_and_prepare_data(seat_data_path, event_data_path, target_game, target_date)
        X_train, X_test, y_class_test, y_reg_test, class_pred, reg_pred, df_processed = optimizer.train_model(df)
        timing_analysis = optimizer.analyze_timing_patterns(df_processed)

        # Store price statistics for comparison
        optimizer.price_stats = {
            'min': float(df['price'].min()),
            'max': float(df['price'].max()),
            'median': float(df['price'].median()),
            '25th': float(df['price'].quantile(0.25)),
            '75th': float(df['price'].quantile(0.75))
        }

        # FIXED: Calculate actual current days until event
        try:
            from datetime import datetime
            event_date = datetime.strptime(target_date, '%Y-%m-%d')
            current_date = datetime.now()
            current_days = (event_date - current_date).days
            current_days = max(1, min(365, current_days))  # Ensure reasonable bounds
        except:
            current_days = 15  # Fallback to 15 days for baseball
        
        print(f"Current days until event: {current_days}")
        # Get consistent timing recommendations
        average_best_days = float(timing_analysis.get('overall_median_days', 14))
        average_best_days = max(3, min(25, average_best_days))  # Baseball-specific bounds
        print(f"Average best days: {average_best_days}")
        print(f"Optimal buy date: {current_days - average_best_days} days from now")

        # Use the most recent ticket listing (closest to today, but before the event) for probability prediction
        df_recent = df[df['days_until_event'] >= 0].copy()
        if not df_recent.empty:
            # Find the row with the minimum days_until_event (closest to today)
            idx = df_recent['days_until_event'].idxmin()
            recent_row = df_recent.loc[idx]
            feature_cols = optimizer.create_features(df_processed).columns
            current_features = np.zeros(len(feature_cols))
            for i, col in enumerate(feature_cols):
                if col in recent_row:
                    current_features[i] = recent_row[col]
            # Use the actual days_until_event for context
            current_days_for_prob = int(recent_row['days_until_event'])
        else:
            # Fallback to median/average if no recent ticket
            avg_price = float(df['price'].median())
            avg_quantity = float(df['quantity'].median())
            feature_cols = optimizer.create_features(df_processed).columns
            current_features = np.zeros(len(feature_cols))
            if 'days_until_event' in feature_cols:
                current_features[feature_cols.get_loc('days_until_event')] = current_days
            if 'price' in feature_cols:
                current_features[feature_cols.get_loc('price')] = avg_price
            if 'quantity' in feature_cols:
                current_features[feature_cols.get_loc('quantity')] = avg_quantity
            if 'day_of_week' in feature_cols:
                current_features[feature_cols.get_loc('day_of_week')] = datetime.now().weekday()
            if 'is_weekend' in feature_cols:
                current_features[feature_cols.get_loc('is_weekend')] = 1 if datetime.now().weekday() >= 5 else 0
            if 'month' in feature_cols:
                current_features[feature_cols.get_loc('month')] = datetime.now().month
            if 'time_trend' in feature_cols:
                current_features[feature_cols.get_loc('time_trend')] = current_days / 365
            if 'price_percentile' in feature_cols:
                price_rank = (avg_price - optimizer.price_stats['min']) / (optimizer.price_stats['max'] - optimizer.price_stats['min'] + 1e-6)
                current_features[feature_cols.get_loc('price_percentile')] = price_rank
            current_days_for_prob = current_days

        # Get actual probability prediction using the most recent ticket listing
        optimal_prob, predicted_price = optimizer.predict_optimal_timing(current_features)

        print(f"Raw optimal probability: {optimal_prob}")
        print(f"Raw predicted price: {predicted_price}")

        # Ensure we have valid values
        optimal_prob = float(optimal_prob) if not pd.isna(optimal_prob) else 0.5
        predicted_price = float(predicted_price) if not pd.isna(predicted_price) else float(df['price'].median())

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

        price_range = [round(price_min, 2), round(price_max, 2)]
        buy_price_range = [round(price_25th, 2), round(price_25th * 1.15, 2)]
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
                    section_25th = float(section_data['price'].quantile(0.25))
                    section_median = float(section_data['price'].median())

                    if not any(pd.isna([section_min, section_max, section_25th, section_median])):
                        price_range_by_section[str(section)] = [round(section_min, 2), round(section_max, 2)]
                        buy_price_by_section[str(section)] = [round(section_25th, 2), round(section_25th * 1.15, 2)]
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

        # --- Recommendation logic based on optimal buy days ---
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

        # Probability context
        prob_percentage = int(optimal_prob * 100)
        prob_context = (
            f"Based on historical data analysis, current market conditions suggest a {prob_percentage}% "
            f"probability that this is a good time to buy tickets. This is calculated using factors like "
            f"current pricing trends, days until the event, and historical patterns for similar baseball games."
        )

        # Ensure predicted price is reasonable
        predicted_price = max(price_min, min(predicted_price, price_max))
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
            'buy_days_overall': round(average_best_days, 1),  # Add this for frontend compatibility
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

        # Fallback optimal days
        average_best_days = 14.0
        days_until_optimal = int(round(current_days - average_best_days))
        if days_until_optimal > 1:
            recommendation = f"We do not recommend buying right now. You should buy in {days_until_optimal} days."
            rec_code = 'yellow'
            optimal_prob = 0.4
        elif days_until_optimal == 1:
            recommendation = "We do not recommend buying right now. You should buy tomorrow."
            rec_code = 'yellow'
            optimal_prob = 0.5
        elif days_until_optimal == 0:
            recommendation = "Now is the optimal time to buy."
            rec_code = 'green'
            optimal_prob = 0.7
        else:  # days_until_optimal < 0
            recommendation = "Optimal time has passed. Buy as soon as possible."
            rec_code = 'yellow'
            optimal_prob = 0.6

        return {
            'event': target_game,
            'event_date': target_date,
            'event_details': {'segment': 'Sports', 'genre': 'Baseball'},
            'optimal_probability': optimal_prob,
            'probability_context': f"Limited data available. Using baseball-specific timing patterns for games {current_days} days away.",
            'predicted_price': round(price_median, 2),
            'predicted_price_range': [round(price_25th, 2), round(price_max, 2)],
            'average_best_days': average_best_days,
            'buy_days_overall': average_best_days,  # Add this for frontend compatibility
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