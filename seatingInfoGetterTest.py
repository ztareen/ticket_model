import requests
import pandas as pd
from datetime import datetime

# ------- Reusable Function -------
def get_filtered_events(client_id, keyword, unwanted_labels=None, min_price=None, max_price=None, per_page=10, page=1):
    url = "https://api.seatgeek.com/2/events"
    params = {
        "q": keyword,
        "client_id": client_id,
        "per_page": per_page,
        "page": page,
        "listing_count.gt": 0
    }

    if min_price is not None:
        params["lowest_price.gte"] = min_price
    if max_price is not None:
        params["highest_price.lte"] = max_price

    response = requests.get(url, params=params)
    response.raise_for_status()
    events = response.json()["events"]

    # Filter out events with missing or zero lowest price
    events = [
        event for event in events
        if event.get("stats", {}).get("lowest_price") not in [None, 0]
    ]

    # Remove unwanted labels
    if unwanted_labels:
        events = [
            event for event in events
            if all(label.lower() not in event['title'].lower() for label in unwanted_labels)
        ]

    df = pd.DataFrame([{
        "Title": event["title"],
        "Venue": event["venue"]["name"],
        "City": event["venue"]["city"],
        "Date": datetime.strptime(event["datetime_local"], "%Y-%m-%dT%H:%M:%S"),
        "Lowest Price": event.get("stats", {}).get("lowest_price"),
        "Highest Price": event.get("stats", {}).get("highest_price"),
        "Listing Count": event.get("stats", {}).get("listing_count"),
        "URL": event["url"]
    } for event in events])

    return df


# -------- Main Execution --------
if __name__ == "__main__":
    client_id = "NTAzNDk2MTZ8MTc0ODAyMzY5MS41NzUwNDY"
    keyword = input("Enter team or keyword to search (e.g. 'Mariners'): ")
    min_price = int(input("Enter minimum price to filter (e.g. 30): "))
    max_price = int(input("Enter maximum price to filter (e.g. 250): "))
    unwanted_labels = ["pinstripe pass", "parking"]

    df = get_filtered_events(client_id, keyword, unwanted_labels, min_price, max_price)
    print("\nFiltered Events:")
    print(df.head(20))  # display first 20 rows