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
        data = request.get_json(force=True)
        event_name = data.get('event_name')
        event_date = data.get('event_date')
        # Construct seat data path based on event name and date
        # Example: Seattle-Mariners-at-Los-Angeles-Angels-2025-07-27.csv
        if event_name and event_date:
            # Always format filename as 'Seattle-Mariners-at-Opponent-YYYY-MM-DD.csv'
            # Extract opponent from event_name (assume format: 'Seattle Mariners at Opponent' or 'Opponent at Seattle Mariners')
            base_team = 'Seattle-Mariners'
            # Remove extra info (e.g., after dash or colon)
            main_name = event_name.split('-')[0].split(':')[0].strip()
            # Try to extract opponent
            opponent = None
            if ' at ' in event_name:
                parts = event_name.split(' at ')
                if base_team.replace('-', ' ') in parts[0]:
                    opponent = parts[1]
                else:
                    opponent = parts[0]
            elif ' vs ' in event_name:
                parts = event_name.split(' vs ')
                if base_team.replace('-', ' ') in parts[0]:
                    opponent = parts[1]
                else:
                    opponent = parts[0]
            else:
                # Fallback: remove 'Seattle Mariners' and use rest
                opponent = event_name.replace('Seattle Mariners', '').strip()
            # Remove anything after a dash, colon, or parenthesis (special event info)
            opponent = opponent.split('-')[0].split(':')[0].split('(')[0].strip()
            # Remove trailing 'vs.' or 'at' if present
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
            optimal_prob = float(optimal_prob)
            predicted_price = float(predicted_price)
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
                timing_by_section = {str(k): round(float(v), 2) for k, v in section_group.items()}
                # Predicted price by section: use mean price per section
                price_group = df.groupby('section')['price'].mean()
                predicted_price_by_section = {str(k): round(float(v), 2) for k, v in price_group.items()}
                # Buy price by section: use quantiles per section
                for section, group in df.groupby('section'):
                    buy_low = round(float(group['price'].quantile(0.10)), 2)
                    buy_high = round(float(group['price'].quantile(0.30)), 2)
                    buy_price_by_section[str(section)] = [buy_low, buy_high]
                    buy_days_by_section[str(section)] = round(float(group['optimal_days_before'].mean()), 2) if 'optimal_days_before' in group.columns else None
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
            wait_days = None
            if optimal_prob > 0.7:
                recommendation = 'Good time to buy!'
                rec_code = 'green'
            elif optimal_prob > 0.4:
                recommendation = 'Consider waiting a bit longer'
                rec_code = 'yellow'
                wait_days = max(0, round(average_best_days - current_features[0], 2))
            else:
                if 'optimal_days_before' in df_with_optimal.columns:
                    future_days = df_with_optimal[df_with_optimal['optimal_days_before'] > 0]['optimal_days_before']
                    if not future_days.empty:
                        wait_days = round(float(future_days.mean() - current_features[0]), 2)
                        wait_days = max(0, wait_days)
                recommendation = 'We recommend waiting {} more days'.format(wait_days if wait_days is not None else '?')
                rec_code = 'red'
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
                'feature_importance': top_features,
                'timing_by_zone': timing_by_zone,
                'timing_by_section': timing_by_section,
                'price_range': price_range,
                'price_range_by_section': price_range_by_section,
                'buy_price_range': buy_price_range,
                'recommendation': recommendation,
                'recommendation_code': rec_code,
                'wait_days': wait_days,
                'predicted_price_by_section': predicted_price_by_section,
                'buy_price_by_section': buy_price_by_section,
                'buy_days_by_section': buy_days_by_section
            })
        else:
            # Fallback to default model if no event_name/date provided
            result = run_ticket_model()
            return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
                # Never show Mariners vs. Miami Marlins on 2025-04-25
                if (
                    (ev.get('event_name', '').lower().find('mariners') != -1 and
                     ev.get('event_name', '').lower().find('miami marlins') != -1 and
                     ev.get('start_date', '').startswith('2025-04-25'))
                ):
                    return False
                # Never show event with start_date exactly '2025-08-01 20:00:00'
                if ev.get('start_date', '') == '2025-08-01 20:00:00':
                    return False
                event_date = datetime.strptime(ev['start_date'][:10], "%Y-%m-%d").date()
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


#optimal time to buy tickets
# wait this many days to buy
# predicted lowest price
# venue of field like what city or stadium
# time of event
# day of week
# rival? if theyre also in teh AL west
# calendar with all the mariners games
# u click on a game and it runs the model for that
# then it shows all the aformetioned stuff
# make a readme
# make a requirements.txt