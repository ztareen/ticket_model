import requests
import numpy as np
import pandas as pd
import json
from datetime import datetime, timezone, timedelta

event_info_list = []
uselessKeywords = [
    "Pinstripe Pass", "Premium Seating", "Group Tickets",
    "Suite", "Club Access",
    "Ballpark Tours", "Pregame Tours", "Non-Gameday"
]

keyword = "seattle_mariners"
start_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
end_utc = (start_utc + timedelta(days=7)).replace(hour=23, minute=59, second=59)

start_utc_str = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
end_utc_str = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

url = (
    f'https://app.ticketmaster.com/discovery/v2/events.json'
    f'?apikey=KntBcmnLD8BBG0DpsV1GMMakFSFcoaof'
    f'&keyword=Seattle%20Mariners'
    f'&startDateTime={start_utc_str}'
    f'&endDateTime={end_utc_str}'
    f'&size=20'
)

response = requests.get(url)

if response.status_code != 200:
    print(f"Request failed with status code {response.status_code}")
    exit()

data = response.json()

try:
    events = data["_embedded"]["events"]
except (KeyError, IndexError) as e:
    print("No event data found or structure changed:", e)
    exit()

for i, event in enumerate(events, start=1):
    name = event["name"]
    if any(keyword.lower() in name.lower() for keyword in uselessKeywords):
        continue  # Skip unwanted variations

    if "sales" not in event:
        continue  # Skip if no sales info

    print(f"Event {i}: {event['name']}")
    print(f" → Official Ticketmaster URL: {event.get('url', 'No URL available')}")  # Print URL for users

    event_info = {
        "name": event["name"],
        "id": event["id"],
        "url": event.get("url", ""),  # Official event URL (Ticketmaster page)
        "start_date": event["dates"]["start"].get("localDate", ""),
        "start_time": event["dates"]["start"].get("localTime", ""),
        "datetime_utc": event["dates"]["start"].get("dateTime", ""),
        "timezone": event["dates"].get("timezone", ""),
        "ticket_status": event["dates"]["status"].get("code", ""),
        "public_sale_start": event["sales"]["public"].get("startDateTime", ""),
        "public_sale_end": event["sales"]["public"].get("endDateTime", ""),
        "presales": [
            {
                "name": presale.get("name", ""),
                "start": presale.get("startDateTime", ""),
                "end": presale.get("endDateTime", "")
            }
            for presale in event["sales"].get("presales", [])
        ],
        "image": event["images"][0]["url"] if event.get("images") else "",
        "info": event.get("info", ""),
        "please_note": event.get("pleaseNote", ""),
        "seatmap_url": event.get("seatmap", {}).get("staticUrl", ""),
        "accessibility_info": event.get("accessibility", {}).get("info", ""),
        "ticket_limit": event.get("ticketLimit", {}).get("info", ""),
        "official_ticket_url": event.get("url", "")  # Add this for clarity & access in dataframe
        #standings of team factored into model
        #playoffs
        #rivalry games
    }
    event_info_list.append(event_info)


# Create DataFrame and save CSV
dataFrameEvents = pd.DataFrame(event_info_list)

today_str = datetime.now().strftime("%Y-%m-%d")
filename = f"event_data_{today_str}.csv"
dataFrameEvents.to_csv(filename, index=False)