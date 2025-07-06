# backend/app.py
from flask import Flask, jsonify
import subprocess
import json

app = Flask(__name__)

@app.route("/mariners-events")
def mariners_events():
    # Run your scripts in order
    subprocess.run(["python", "ticketmaster_eventgetter.py"])
    subprocess.run(["python", "getCSV.py"])
    subprocess.run(["python", "developFinalInitial.py"])

    # Load final result (assuming you write a JSON output)
    with open("final_events.json", "r") as f:
        events = json.load(f)

    return jsonify({"events": events})

if __name__ == "__main__":
    app.run(debug=True)