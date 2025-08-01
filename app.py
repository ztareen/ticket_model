from flask import Flask, jsonify, render_template, request
from ticket_model import run_ticket_model
from flask_cors import CORS
import json
import os
from datetime import datetime
from ticketmaster_eventgetter import event_info

app = Flask(__name__)
CORS(app)

@app.route("/")  # <-- This is the route you were missing
def home():
    return render_template("index.html")

@app.route("/api/run-model", methods=["GET"])
def run_model():
    try:
        result = run_ticket_model()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# New endpoint for POST /run
@app.route('/run', methods=['POST'])
def run_script():
    try:
        data = request.get_json()
        event_name = data.get('event_name', '')
        event_date = data.get('event_date', '')
        
        # Block the specific event from ever being processed
        if (event_name and event_date and 
            'mariners' in event_name.lower() and 
            'miami marlins' in event_name.lower() and 
            '2025-04-25' in event_date):
            return jsonify({'error': 'This event is not available for analysis.'}), 400
        
        if event_name and event_date:
            # Extract opponent name from event name
            opponent = event_name
            
            # Handle different event name formats
            if ' vs. ' in event_name:
                parts = event_name.split(' vs. ')
                if len(parts) == 2:
                    # If first part is "Athletics" and second part is "Seattle Mariners", take "Athletics"
                    if parts[0].strip() == 'Athletics' and 'Seattle Mariners' in parts[1]:
                        opponent = 'Athletics'
                    # If first part is "Seattle Mariners", take the second part
                    elif 'Seattle Mariners' in parts[0]:
                        opponent = parts[1].strip()
                    # Otherwise take the first part
                    else:
                        opponent = parts[0].strip()
            elif ' vs ' in event_name:
                parts = event_name.split(' vs ')
                if len(parts) == 2:
                    if parts[0].strip() == 'Athletics' and 'Seattle Mariners' in parts[1]:
                        opponent = 'Athletics'
                    elif 'Seattle Mariners' in parts[0]:
                        opponent = parts[1].strip()
                    else:
                        opponent = parts[0].strip()
            elif ' at ' in event_name:
                parts = event_name.split(' at ')
                if len(parts) == 2:
                    if parts[0].strip() == 'Athletics' and 'Seattle Mariners' in parts[1]:
                        opponent = 'Athletics'
                    elif 'Seattle Mariners' in parts[0]:
                        opponent = parts[1].strip()
                    else:
                        opponent = parts[0].strip()
            
            # Clean up the opponent name
            opponent = opponent.replace('vs.', '').replace('at', '').replace('vs', '').strip()
            # Replace spaces with dashes
            opponent = opponent.replace(' ', '-')
            seat_data_filename = f"Seattle-Mariners-vs-{opponent}-{event_date}.csv"
            seat_data_dir = r"C:\Users\zarak\Downloads\TestData_Mariners"
            seat_data_path = os.path.join(seat_data_dir, seat_data_filename)
            # Use today's event data file
            today = datetime.now().date()
            event_data_path = f"C:/Users/zarak/OneDrive/Documents/GitHub/ticket_model/event_data_{today.year}.{today.month:02d}.{today.day:02d}.csv"
            # Run the model for this event
            from ticket_model import TicketTimingOptimizer
            optimizer = TicketTimingOptimizer()
            # Use event_name and event_date as target_game and target_date
            df = optimizer.load_and_prepare_data(seat_data_path, event_data_path, event_name, event_date)
            X_train, X_test, y_class_test, y_reg_test, class_pred, reg_pred, df_with_optimal = optimizer.train_model(df)
            timing_analysis = optimizer.analyze_timing_patterns(df_with_optimal)
            current_features = [
                30, 150, 4, 2, 0, 25, 6, 0.05, 145, 5, 0.1, 0.02, 50
            ]
            n_features = len(optimizer.create_features(df).columns)
            import numpy as np
            current_features = np.array(current_features)
            if len(current_features) < n_features:
                current_features = np.concatenate([current_features, np.zeros(n_features - len(current_features))])
            optimal_prob, predicted_price = optimizer.predict_optimal_timing(current_features)
            # Ensure optimal_prob is always a valid float between 0 and 1
            try:
                optimal_prob = float(optimal_prob)
                if not (0.0 <= optimal_prob <= 1.0) or (optimal_prob is None) or (optimal_prob != optimal_prob):
                    optimal_prob = 0.0
            except Exception:
                optimal_prob = 0.0
            try:
                predicted_price = float(predicted_price)
            except Exception:
                predicted_price = 0.0
            average_best_days = float(timing_analysis['overall_avg_days'])
            if optimizer.feature_importance is not None:
                top_features = optimizer.feature_importance.head(3)['feature'].tolist()
            else:
                top_features = []
            timing_by_zone = {str(k): round(float(v), 2) for k, v in timing_analysis['by_zone'].items()}
            timing_by_section = {}
            predicted_price_by_section = {}
            buy_price_by_section = {}
            buy_days_by_section = {}
            if 'section' in df.columns and 'optimal_days_before' in df_with_optimal.columns:
                section_group = df_with_optimal.groupby('section')['optimal_days_before'].mean()
                # Remove NaN values from groupby result
                section_group = section_group.dropna()
                timing_by_section = {str(k): round(float(v), 2) for k, v in section_group.items()}
                buy_days_by_section = timing_by_section.copy()
                # Predicted price by section: use mean price per section
                price_group = df.groupby('section')['price'].mean()
                predicted_price_by_section = {str(k): round(float(v), 2) for k, v in price_group.items()}
                # Buy price by section: use quantiles per section
                for section, group in df.groupby('section'):
                    buy_low = round(float(group['price'].quantile(0.10)), 2)
                    buy_high = round(float(group['price'].quantile(0.30)), 2)
                    buy_price_by_section[str(section)] = [buy_low, buy_high]
            else:
                # Fallback: if columns missing, set None for each section
                if 'section' in df.columns:
                    buy_days_by_section = {str(k): None for k in df['section'].unique()}
                else:
                    buy_days_by_section = {}
            price_min = float(df['price'].min())
            price_max = float(df['price'].max())
            price_range = [round(price_min, 2), round(price_max, 2)]
            price_range_by_section = {}
            if 'section' in df.columns:
                for section, group in df.groupby('section'):
                    min_p = round(float(group['price'].min()), 2)
                    max_p = round(float(group['price'].max()), 2)
                    price_range_by_section[str(section)] = [min_p, max_p]
            buy_price_low = round(float(df['price'].quantile(0.10)), 2)
            buy_price_high = round(float(df['price'].quantile(0.30)), 2)
            buy_price_range = [buy_price_low, buy_price_high]
            pred_price_low = round(float(df['price'].quantile(0.10)), 2)
            pred_price_high = round(float(df['price'].quantile(0.90)), 2)
            predicted_price_range = [pred_price_low, pred_price_high]
            # Calculate if optimal time has passed
            current_days_until_event = current_features[0] if len(current_features) > 0 else 30
            optimal_time_has_passed = current_days_until_event > average_best_days
            
            if optimal_time_has_passed:
                # If optimal time has passed, adjust recommendation based on how far past we are
                days_past_optimal = current_days_until_event - average_best_days
                if days_past_optimal <= 7:
                    recommendation = 'We recommend buying now - optimal time recently passed!'
                    rec_code = 'green'
                elif days_past_optimal <= 14:
                    recommendation = 'We recommend buying now - optimal time has passed'
                    rec_code = 'yellow'
                else:
                    recommendation = 'We recommend buying now - prices may be higher'
                    rec_code = 'yellow'
            else:
                # If optimal time is in the future
                if optimal_prob > 0.7:
                    recommendation = 'We recommend buying now!'
                    rec_code = 'green'
                elif optimal_prob > 0.4:
                    recommendation = 'We recommend waiting'
                    rec_code = 'yellow'
                else:
                    recommendation = 'We recommend not purchasing at all'
                    rec_code = 'yellow'
            event_details = {}
            if not df.empty:
                for col in ['venue_id', 'segment', 'genre', 'ticket_status']:
                    if col in df.columns:
                        event_details[col] = str(df.iloc[0][col])
            prob_context = (
                "This is the estimated probability that buying now is the optimal time. "
                "A value close to 1 means it's likely the best time to buy, while a value near 0 means it's likely better to wait."
            )
            return jsonify({
                'event': event_name,
                'event_date': event_date,
                'event_details': event_details,
                'optimal_probability': round(optimal_prob, 2),
                'probability_context': prob_context,
                'predicted_price': round(predicted_price, 2),
                'predicted_price_range': predicted_price_range,
                'average_best_days': round(average_best_days, 2),
                'buy_days_overall': round(average_best_days, 2),
                'feature_importance': top_features,
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
            })
        else:
            # Fallback to default model if no event_name/date provided
            result = run_ticket_model()
            return jsonify(result)
    except Exception as e:
        error_msg = str(e)
        if "Seat data file not found" in error_msg:
            return jsonify({'error': f"Data file not found for this event. Please ensure the seat data file exists: {error_msg}"}), 500
        elif "Error reading seat data file" in error_msg:
            return jsonify({'error': f"Error reading data file: {error_msg}"}), 500
        elif "Error parsing dates" in error_msg:
            return jsonify({'error': f"Date format error in data file: {error_msg}"}), 500
        elif "No valid data found" in error_msg:
            return jsonify({'error': f"No valid data found in the seat data file: {error_msg}"}), 500
        else:
            return jsonify({'error': f"Model execution failed: {error_msg}"}), 500

# New endpoint to serve event info from ticketmaster_eventgetter
@app.route("/api/events", methods=["GET"])
def get_events():
    try:
        # Ensure event_info is a list
        events = event_info if isinstance(event_info, list) else [event_info]
        # Filter for upcoming events only (from today onward)
        today = datetime.now().date()
        def is_upcoming(ev):
            try:
                if not ev.get('start_date'): return False
                
                # Comprehensive filtering for the blocked event
                event_name = ev.get('event_name', '').lower()
                start_date = ev.get('start_date', '')
                
                # Block Mariners vs. Miami Marlins on 2025-04-25 (multiple variations)
                # Check for the specific event name regardless of date format
                if ('mariners' in event_name and 
                    ('miami marlins' in event_name or 'marlins' in event_name) and 
                    ('2025-04-25' in start_date or '4/25/25' in event_name or '4/25/2025' in event_name)):
                    return False
                
                # Block any event with this specific date and team combination
                if ('2025-04-25' in start_date and 
                    'mariners' in event_name and 
                    'miami' in event_name):
                    return False
                
                # Block the specific event by name pattern regardless of date
                if ('4/25/25' in event_name and 
                    'mariners' in event_name and 
                    ('miami' in event_name or 'marlins' in event_name)):
                    return False
                
                # Never show event with start_date exactly '2025-08-01 20:00:00'
                if start_date == '2025-08-01 20:00:00':
                    return False
                
                event_date = datetime.strptime(start_date[:10], "%Y-%m-%d").date()
                return event_date >= today
            except Exception:
                return False
        upcoming_events = list(filter(is_upcoming, events))
        # Sort by date ascending
        upcoming_events.sort(key=lambda ev: ev.get('start_date', ''))
        return jsonify(upcoming_events)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)