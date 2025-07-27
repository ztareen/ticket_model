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
from datetime import date
warnings.filterwarnings('ignore')#test

today = date.today()
# NOTE: seat_data_path is now always passed as a function argument from app.py
# Remove hardcoded SEAT_DATA_PATH and ensure all functions use the provided path
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
    # Example usage: pass seat_data_path as argument
    # seat_data_path, event_data_path, target_game, target_date should be provided by caller
    # ...existing code...
    return optimizer, None, None

if __name__ == "__main__":
    # For manual testing, you can specify a seat_data_path here
    pass

def run_ticket_model(seat_data_path, event_data_path, target_game, target_date):
    # Called by app.py with all arguments provided
    optimizer = TicketTimingOptimizer()
    df = optimizer.load_and_prepare_data(seat_data_path, event_data_path, target_game, target_date)
    X_train, X_test, y_class_test, y_reg_test, class_pred, reg_pred, df_with_optimal = optimizer.train_model(df)
    timing_analysis = optimizer.analyze_timing_patterns(df_with_optimal)

    current_features = np.array([
        30, 150, 4, 2, 0, 25, 6, 0.05, 145, 5, 0.1, 0.02, 50
    ])
    n_features = len(optimizer.create_features(df).columns)
    if len(current_features) < n_features:
        current_features = np.concatenate([current_features, np.zeros(n_features - len(current_features))])

    optimal_prob, predicted_price = optimizer.predict_optimal_timing(current_features)

    # Convert NumPy types to native Python types
    optimal_prob = float(optimal_prob)
    predicted_price = float(predicted_price)
    average_best_days = float(timing_analysis['overall_avg_days'])

    # Collect feature importance (top 3)
    if optimizer.feature_importance is not None:
        top_features = optimizer.feature_importance.head(3)['feature'].tolist()
    else:
        top_features = []

    # Collect optimal timing by zone and by section
    timing_by_zone = {str(k): round(float(v), 2) for k, v in timing_analysis['by_zone'].items()}
    timing_by_section = {}
    buy_days_by_section = {}
    if 'section' in df.columns and 'optimal_days_before' in df_with_optimal.columns:
        section_group = df_with_optimal.groupby('section')['optimal_days_before'].mean()
        timing_by_section = {str(k): round(float(v), 2) for k, v in section_group.items()}
        buy_days_by_section = timing_by_section.copy()
    # Overall event buy days
    buy_days_overall = round(float(timing_analysis['overall_avg_days']), 2)

    # Group sections for frontend display
    section_group_map = {
        'infield': [],
        'outfield': [],
        'bleachers': [],
        'suite': [],
        'other': []
    }
    for section, days in timing_by_section.items():
        section_lower = section.lower()
        if 'infield' in section_lower:
            section_group_map['infield'].append(days)
        elif 'outfield' in section_lower:
            section_group_map['outfield'].append(days)
        elif 'bleacher' in section_lower:
            section_group_map['bleachers'].append(days)
        elif 'suite' in section_lower:
            section_group_map['suite'].append(days)
        else:
            section_group_map['other'].append(days)
    # Calculate average for each group (if any sections in group)
    timing_by_section_group = {}
    for group, values in section_group_map.items():
        if values:
            timing_by_section_group[group] = round(float(np.mean(values)), 2)

    # Collect price range (overall and by section)
    price_min = float(df['price'].min())
    price_max = float(df['price'].max())
    price_range = [round(price_min, 2), round(price_max, 2)]
    price_range_by_section = {}
    predicted_price_by_section = {}
    buy_price_by_section = {}
    if 'section' in df.columns:
        for section, group in df.groupby('section'):
            min_p = round(float(group['price'].min()), 2)
            max_p = round(float(group['price'].max()), 2)
            price_range_by_section[str(section)] = [min_p, max_p]
            # Predicted price for this section: use model_regressor on mean features for section
            section_features = optimizer.create_features(group).mean().values
            section_features = np.array(section_features)
            if len(section_features) < n_features:
                section_features = np.concatenate([section_features, np.zeros(n_features - len(section_features))])
            try:
                section_pred_price = optimizer.model_regressor.predict(section_features.reshape(1, -1))[0]
                predicted_price_by_section[str(section)] = round(float(section_pred_price), 2)
            except Exception:
                predicted_price_by_section[str(section)] = None
            # Recommended buy price range for this section (10th-30th percentile)
            buy_low = round(float(group['price'].quantile(0.10)), 2)
            buy_high = round(float(group['price'].quantile(0.30)), 2)
            buy_price_by_section[str(section)] = [buy_low, buy_high]

    # Recommended buy price range (e.g., 10th-30th percentile of prices)
    buy_price_low = round(float(df['price'].quantile(0.10)), 2)
    buy_price_high = round(float(df['price'].quantile(0.30)), 2)
    buy_price_range = [buy_price_low, buy_price_high]

    # Expanded predicted price range (e.g., 10th-90th percentile)
    pred_price_low = round(float(df['price'].quantile(0.10)), 2)
    pred_price_high = round(float(df['price'].quantile(0.90)), 2)
    predicted_price_range = [pred_price_low, pred_price_high]

    # Recommendation logic and wait days
    wait_days = None
    if optimal_prob > 0.7:
        recommendation = 'Good time to buy!'
        rec_code = 'green'
    elif optimal_prob > 0.4:
        recommendation = 'Consider waiting a bit longer'
        rec_code = 'yellow'
        # Suggest wait days as the difference between now and average best days
        wait_days = max(0, round(average_best_days - current_features[0], 2))
    else:
        # Find the most common optimal_days_before (mode) or average
        if 'optimal_days_before' in df_with_optimal.columns:
            # Only consider future days
            future_days = df_with_optimal[df_with_optimal['optimal_days_before'] > 0]['optimal_days_before']
            if not future_days.empty:
                wait_days = round(float(future_days.mean() - current_features[0]), 2)
                wait_days = max(0, wait_days)
        recommendation = 'We recommend waiting {} more days'.format(wait_days if wait_days is not None else '?')
        rec_code = 'red'

    # Event details from event metadata (grab from df if available)
    event_details = {}
    if not df.empty:
        for col in ['venue_id', 'segment', 'genre', 'ticket_status']:
            if col in df.columns:
                event_details[col] = str(df.iloc[0][col])

    # Probability context
    prob_context = (
        "This is the estimated probability that buying now is the optimal time. "
        "A value close to 1 means it's likely the best time to buy, while a value near 0 means it's likely better to wait."
    )

    return {
        'event': target_game,
        'event_date': target_date,
        'event_details': event_details,
        'optimal_probability': round(optimal_prob, 2),
        'probability_context': prob_context,
        'predicted_price': round(predicted_price, 2),
        'predicted_price_range': predicted_price_range,
        'average_best_days': round(average_best_days, 2),
        'feature_importance': top_features,
        'timing_by_zone': timing_by_zone,
        'timing_by_section': timing_by_section,
        'price_range': price_range,
        'price_range_by_section': price_range_by_section,
        'buy_price_range': buy_price_range,
        'predicted_price_by_section': predicted_price_by_section,
        'buy_price_by_section': buy_price_by_section,
        'buy_days_by_section': buy_days_by_section,
        'buy_days_overall': buy_days_overall,
        'recommendation': recommendation,
        'recommendation_code': rec_code,
        'wait_days': wait_days
    }