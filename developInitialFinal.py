import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import math
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')#test

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
        with open(seat_data_path, 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            
            for row in csvreader:
                seat_data.append({
                    'date': row[0],
                    'zone': row[1],
                    'section': row[2],
                    'row': row[3],
                    'quantity': int(row[4]),
                    'price': float(row[5])
                })
        
        # Load event metadata
        event_metadata = {}
        with open(event_data_path, 'r', encoding='utf-8') as file:
            csvreader = csv.DictReader(file)
            for row in csvreader:
                if (row["start_date"] == target_date or 
                    target_date in row.get("start_date", "") or
                    target_game in row.get("event_name", "")):
                    event_metadata = row
                    break
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(seat_data)
        
        # Parse dates and calculate time features
        df['date'] = pd.to_datetime(df['date'])
        event_date = pd.to_datetime(target_date)
        
        # Calculate days until event (key feature for timing)
        df['days_until_event'] = (event_date - df['date']).dt.days
        
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
            # Add default values if no event metadata found
            df['venue_id'] = 0
            df['segment'] = 'Unknown'
            df['genre'] = 'Unknown'
            df['ticket_status'] = 'Unknown'
        
        # Create price trend features - fix the rolling window calculations
        df = df.sort_values(['zone', 'section', 'row', 'date'])
        
        # Calculate price changes and trends using a safer approach
        df['price_change'] = df.groupby(['zone', 'section', 'row'])['price'].pct_change()
        
        # Fix rolling calculations by using transform instead of apply
        df['price_rolling_mean_3d'] = df.groupby(['zone', 'section', 'row'])['price'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['price_rolling_std_3d'] = df.groupby(['zone', 'section', 'row'])['price'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        )
        
        # Calculate price volatility (standard deviation of price changes)
        df['price_volatility'] = df.groupby(['zone', 'section', 'row'])['price_change'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        )
        
        # Quantity-based features
        df['quantity_change'] = df.groupby(['zone', 'section', 'row'])['quantity'].pct_change()
        df['total_quantity_by_section'] = df.groupby(['section', 'date'])['quantity'].transform('sum')
        
        # Create target variable: future price minimum
        # For each observation, find the minimum price that will occur in the next 7 days
        df['future_min_price'] = df.groupby(['zone', 'section', 'row'])['price'].transform(
            lambda x: x.rolling(window=7, min_periods=1).min().shift(-6)
        )
        
        # Create binary target: will price drop significantly in next 7 days?
        df['price_will_drop'] = ((df['future_min_price'] / df['price']) < 0.95).astype(int)
        
        # Fill NaN and inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def create_features(self, df):
        """Create feature matrix for training"""
        
        # Encode categorical variables
        categorical_cols = ['zone', 'section', 'segment', 'genre', 'ticket_status']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Select features for the model
        feature_cols = [
            'days_until_event', 'price', 'quantity', 'day_of_week', 'is_weekend',
            'week_of_year', 'month', 'price_change', 'price_rolling_mean_3d',
            'price_rolling_std_3d', 'price_volatility', 'quantity_change',
            'total_quantity_by_section'
        ]
        
        # Add encoded categorical features
        for col in categorical_cols:
            if f'{col}_encoded' in df.columns:
                feature_cols.append(f'{col}_encoded')
        
        # Filter existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        return df[feature_cols]
    
    def train_model(self, df):
        """Train XGBoost model to predict optimal buying timing"""
        
        # Prepare features
        X = self.create_features(df)
        
        # Create target: days until event when price is at minimum
        # For each seat, find the day when price was lowest
        price_min_days = df.groupby(['zone', 'section', 'row']).apply(
            lambda group: group.loc[group['price'].idxmin(), 'days_until_event']
        ).reset_index(name='optimal_days_before')
        
        # Merge back with original data
        df_with_optimal = df.merge(
            price_min_days, 
            on=['zone', 'section', 'row'], 
            how='left'
        )
        
        # Create binary classification target: is this the optimal time to buy?
        y_classification = (df_with_optimal['days_until_event'] == df_with_optimal['optimal_days_before']).astype(int)
        
        # Create regression target: price prediction
        y_regression = df['price']
        
        # Split data
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_classification, y_regression, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classification model (optimal timing)
        self.model_classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.model_classifier.fit(X_train_scaled, y_class_train)
        
        # Train regression model (price prediction)
        self.model_regressor = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.model_regressor.fit(X_train_scaled, y_reg_train)
        
        # Evaluate models
        class_pred = self.model_classifier.predict(X_test_scaled)
        reg_pred = self.model_regressor.predict(X_test_scaled)
        
        print("=== MODEL PERFORMANCE ===")
        print(f"Classification Accuracy: {np.mean(class_pred == y_class_test):.3f}")
        print(f"Regression R²: {r2_score(y_reg_test, reg_pred):.3f}")
        print(f"Regression MAE: ${mean_absolute_error(y_reg_test, reg_pred):.2f}")
        
        # Feature importance - initialize properly
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance_class': self.model_classifier.feature_importances_,
            'importance_reg': self.model_regressor.feature_importances_
        }).sort_values('importance_class', ascending=False)
        
        return X_train_scaled, X_test_scaled, y_class_test, y_reg_test, class_pred, reg_pred, df_with_optimal
    
    def predict_optimal_timing(self, current_features):
        """Predict optimal timing for ticket purchase"""
        
        if self.model_classifier is None or self.model_regressor is None:
            raise ValueError("Models not trained yet!")
        
        # Scale features
        current_features_scaled = self.scaler.transform(current_features.reshape(1, -1))
        
        # Get predictions
        optimal_prob = self.model_classifier.predict_proba(current_features_scaled)[0][1]
        predicted_price = self.model_regressor.predict(current_features_scaled)[0]
        
        return optimal_prob, predicted_price
    
    def analyze_timing_patterns(self, df_with_optimal):
        """Analyze patterns in optimal timing"""
        
        # Calculate average optimal timing by different factors
        timing_analysis = {
            'overall_avg_days': df_with_optimal['optimal_days_before'].mean(),
            'by_zone': df_with_optimal.groupby('zone')['optimal_days_before'].mean().to_dict(),
            'by_month': df_with_optimal.groupby('month')['optimal_days_before'].mean().to_dict(),
            'by_price_range': df_with_optimal.groupby(pd.cut(df_with_optimal['price'], bins=5))['optimal_days_before'].mean().to_dict()
        }
        
        return timing_analysis
    
    def visualize_results(self, df, df_with_optimal):
        """Create visualizations of the analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Price trends over time
        axes[0, 0].scatter(df['days_until_event'], df['price'], alpha=0.6, s=10)
        axes[0, 0].set_xlabel('Days Until Event')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].set_title('Price vs Days Until Event')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Optimal timing distribution
        axes[0, 1].hist(df_with_optimal['optimal_days_before'].dropna(), bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Optimal Days Before Event')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Optimal Purchase Timing')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature importance - check if feature_importance exists
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[1, 0].barh(top_features['feature'], top_features['importance_class'])
            axes[1, 0].set_xlabel('Feature Importance')
            axes[1, 0].set_title('Top 10 Features for Timing Prediction')
        else:
            axes[1, 0].text(0.5, 0.5, 'Feature importance not available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Importance')
        
        # 4. Price by zone and timing
        zone_timing = df_with_optimal.groupby(['zone', 'optimal_days_before'])['price'].mean().reset_index()
        for zone in zone_timing['zone'].unique():
            zone_data = zone_timing[zone_timing['zone'] == zone]
            axes[1, 1].plot(zone_data['optimal_days_before'], zone_data['price'], 
                          marker='o', label=f'Zone {zone}', linewidth=2)
        
        axes[1, 1].set_xlabel('Optimal Days Before Event')
        axes[1, 1].set_ylabel('Average Price ($)')
        axes[1, 1].set_title('Average Price by Zone and Timing')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Example usage and main execution
def main():
    # Initialize the optimizer
    optimizer = TicketTimingOptimizer()
    
    # Define file paths and target event
    seat_data_path = "C:/Users/zarak/Downloads/TestData_Mariners/Seattle_Mariners_at_Minnesota_Twins_2025-06-25.csv"
    event_data_path = "C:/Users/zarak/OneDrive/Documents/GitHub/ticket_model/event_data_2025.06.24.csv"
    target_game = "Seattle_Mariners_at_Minnesota_Twins"
    target_date = "2025-06-25"
    
    print("Loading and preparing data...")
    df = optimizer.load_and_prepare_data(seat_data_path, event_data_path, target_game, target_date)
    
    print(f"Loaded {len(df)} ticket records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Days until event range: {df['days_until_event'].min()} to {df['days_until_event'].max()}")
    
    print("\nTraining models...")
    X_train, X_test, y_class_test, y_reg_test, class_pred, reg_pred, df_with_optimal = optimizer.train_model(df)
    
    print("\n=== FEATURE IMPORTANCE ===")
    if optimizer.feature_importance is not None:
        print(optimizer.feature_importance.head(10))
    else:
        print("Feature importance not available")
    
    print("\n=== TIMING ANALYSIS ===")
    timing_analysis = optimizer.analyze_timing_patterns(df_with_optimal)
    print(f"Average optimal timing: {timing_analysis['overall_avg_days']:.1f} days before event")
    print("Optimal timing by zone:")
    for zone, days in timing_analysis['by_zone'].items():
        print(f"  Zone {zone}: {days:.1f} days")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    fig = optimizer.visualize_results(df, df_with_optimal)
    
    # Example prediction for current conditions
    print("\n=== EXAMPLE PREDICTION ===")
    # Simulate current ticket conditions
    current_features = np.array([
        30,  # days_until_event
        150, # price
        4,   # quantity
        2,   # day_of_week
        0,   # is_weekend
        25,  # week_of_year
        6,   # month
        0.05, # price_change
        145,  # price_rolling_mean_3d
        5,    # price_rolling_std_3d
        0.1,  # price_volatility
        0.02, # quantity_change
        50    # total_quantity_by_section
    ])
    
    # Add zeros for categorical encoded features if they exist
    n_features = len(optimizer.create_features(df).columns)
    if len(current_features) < n_features:
        current_features = np.concatenate([current_features, np.zeros(n_features - len(current_features))])
    
    try:
        optimal_prob, predicted_price = optimizer.predict_optimal_timing(current_features)
        print(f"Probability this is optimal timing: {optimal_prob:.3f}")
        print(f"Predicted price: ${predicted_price:.2f}")
        if optimal_prob > 0.7:
            print("🟢 RECOMMENDATION: Good time to buy!")
        elif optimal_prob > 0.4:
            print("🟡 RECOMMENDATION: Consider waiting a bit longer")
        else:
            print("🔴 RECOMMENDATION: Wait for better timing")
            
    except Exception as e:
        print(f"Prediction error: {e}")
    
    print("\n=== SUMMARY INSIGHTS ===")
    print(f"• Best average timing: {timing_analysis['overall_avg_days']:.1f} days before the event")
    print(f"• Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    if optimizer.feature_importance is not None:
        print(f"• Most important factors: {', '.join(optimizer.feature_importance.head(3)['feature'].tolist())}")
    else:
        print("• Most important factors: Not available")
    
    return optimizer, df, df_with_optimal

if __name__ == "__main__":
    optimizer, df, df_with_optimal = main()