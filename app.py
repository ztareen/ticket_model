from flask import Flask, jsonify, render_template
from ticket_model import run_ticket_model
from flask_cors import CORS

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
from flask import request

@app.route('/run', methods=['POST'])
def run_script():
    try:
        result = run_ticket_model()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
