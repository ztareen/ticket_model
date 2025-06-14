import requests
import pandas as pd
from datetime import datetime

def get_filtered_events(client_id, keyword, unwanted_labels=None, min_price=None, max_price=None, per_page=100, page=1):
    url = "https://api.seatgeek.com/2/events"
    params = {
        "q": keyword,
        "client_id": client_id,
        "per_page": per_page,
        "page": page,
        "listing_count.gt": 0  # Only return events with listings
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    events = response.json()["events"]
    print(f"Total events returned: {len(events)}")

    filtered_events = []
    for event in events:
        title = event["title"]
        stats = event.get("stats", {})
        lowest = stats.get("lowest_price")
        highest = stats.get("highest_price")

        if unwanted_labels and any(label.lower() in title.lower() for label in unwanted_labels):
            continue  # Skip if title contains unwanted label

        # Skip if no price info
        if lowest is None or highest is None:
            continue

        # Filter on price range
        if (min_price is not None and lowest < min_price) or (max_price is not None and highest > max_price):
            continue

        filtered_events.append({
            "Title": title,
            "Venue": event["venue"]["name"],
            "City": event["venue"]["city"],
            "Date": datetime.strptime(event["datetime_local"], "%Y-%m-%dT%H:%M:%S"),
            "Lowest Price": lowest,
            "Highest Price": highest,
            "Listing Count": stats.get("listing_count"),
            "URL": event["url"]
        })

    return pd.DataFrame(filtered_events)