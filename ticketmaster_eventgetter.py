import requests
import numpy as np
import pandas as pd
import json
from datetime import datetime, timezone, timedelta

event_info_list = []

uselessKeywords = [
    "Pinstripe Pass", "Premium Seating", "Group Tickets",
    "Suite", "Club Access", "Ballpark Tours", "Pregame Tours", "Non-Gameday", "Gift Card", "Ballpark Tour", "Tours",
]

keyword = "seattle_mariners"
start_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
end_utc = (start_utc + timedelta(days=365)).replace(hour=23, minute=59, second=59)  # Increased to 365 days

start_utc_str = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
end_utc_str = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

# Function to fetch events with pagination
def fetch_all_events():
    all_events = []
    page = 0
    size = 200  # Maximum size per request
    
    while True:
        url = (
            f'https://app.ticketmaster.com/discovery/v2/events.json'
            f'?apikey=KntBcmnLD8BBG0DpsV1GMMakFSFcoaof'
            f'&keyword=Seattle%20Mariners'
            f'&startDateTime={start_utc_str}'
            f'&endDateTime={end_utc_str}'
            f'&size={size}'
            f'&page={page}'
        )
        
        print(f"Fetching page {page}...")
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Request failed with status code {response.status_code}")
            break
            
        data = response.json()
        
        try:
            events = data["_embedded"]["events"]
            all_events.extend(events)
            print(f"Found {len(events)} events on page {page}")
            
            # Check if there are more pages
            page_info = data.get("page", {})
            total_pages = page_info.get("totalPages", 1)
            
            if page >= total_pages - 1:  # Pages are 0-indexed
                break
                
            page += 1
            
        except (KeyError, IndexError) as e:
            print(f"No more event data found on page {page}: {e}")
            break
    
    return all_events

# Fetch all events
print("Starting to fetch all events...")
all_events = fetch_all_events()
print(f"Total events fetched: {len(all_events)}")

# Process each event
for i, event in enumerate(all_events, start=1):
    name = event["name"]
    event_id = event["id"]

    if any(kw.lower() in name.lower() for kw in uselessKeywords):
        continue

    print(f"Processing Event {i}: {name}")

    venue = event.get("_embedded", {}).get("venues", [{}])[0]

    event_info = {
        "event_id": event_id,
        "event_name": name,
        "url": event.get("url", ""),
        "start_date": event.get("dates", {}).get("start", {}).get("localDate", ""),
        "start_time": event.get("dates", {}).get("start", {}).get("localTime", ""),
        "datetime_utc": event.get("dates", {}).get("start", {}).get("dateTime", ""),
        "timezone": event.get("dates", {}).get("timezone", ""),
        "ticket_status": event.get("dates", {}).get("status", {}).get("code", ""),
        "public_sale_start": event.get("sales", {}).get("public", {}).get("startDateTime", ""),
        "public_sale_end": event.get("sales", {}).get("public", {}).get("endDateTime", ""),
        "presales": json.dumps([
            {
                "name": p.get("name", ""),
                "start": p.get("startDateTime", ""),
                "end": p.get("endDateTime", "")
            } for p in event.get("sales", {}).get("presales", [])
        ]),
        "image": event["images"][0]["url"] if event.get("images") else "",
        "info": event.get("info", ""),
        "please_note": event.get("pleaseNote", ""),
        "seatmap_url": event.get("seatmap", {}).get("staticUrl", ""),
        "accessibility_info": event.get("accessibility", {}).get("info", ""),
        "ticket_limit": event.get("ticketLimit", {}).get("info", ""),

        # Venue Info
        "venue_id": venue.get("id", ""),
        "venue_name": venue.get("name", ""),
        "city": venue.get("city", {}).get("name", ""),
        "state": venue.get("state", {}).get("name", ""),
        "country": venue.get("country", {}).get("name", ""),
        "venue_timezone": venue.get("timezone", ""),

        # Classification
        "segment": event.get("classifications", [{}])[0].get("segment", {}).get("name", "") if event.get("classifications") else "",
        "genre": event.get("classifications", [{}])[0].get("genre", {}).get("name", "") if event.get("classifications") else "",
        "subGenre": event.get("classifications", [{}])[0].get("subGenre", {}).get("name", "") if event.get("classifications") else "",
    }

    url_EventDetail = (
    f'https://app.ticketmaster.com/discovery/v2/events/{event_id}.json?apikey=KntBcmnLD8BBG0DpsV1GMMakFSFcoaof'
    f'&id={event_id}'
    )

    detail_resp = requests.get(url_EventDetail)
    if detail_resp.status_code != 200:
        print(f"Detail fetch failed for {event_id}")
        continue

    detail_data = detail_resp.json()

    # Safely pull optional or nested fields
    event_info.update({
        "type": detail_data.get("type", ""),
        "location": json.dumps(detail_data.get("location", {})),
        "name": detail_data.get("name", ""),
        "dates": json.dumps(detail_data.get("dates", {})),
        "sales": json.dumps(detail_data.get("sales", {})),
        "priceRanges": json.dumps(detail_data.get("priceRanges", [])),
        "promoter": json.dumps(detail_data.get("promoter", {})),
        "promoters": json.dumps(detail_data.get("promoters", [])),
        "seatmap": json.dumps(detail_data.get("seatmap", {})),
        "accessibility": json.dumps(detail_data.get("accessibility", {})),
        "ticketLimit": json.dumps(detail_data.get("ticketLimit", {})),
        "classifications": json.dumps(detail_data.get("classifications", [])),
        "externalLinks": json.dumps(detail_data.get("externalLinks", {})),
    })

    event_info_list.append(event_info)

# Create DataFrame
df = pd.DataFrame(event_info_list)

# Sort by date
def parse_datetime_for_sorting(row):
    """Create a datetime object for sorting, handling missing values"""
    if pd.isna(row['start_date']) or row['start_date'] == '':
        # If no start_date, try to use datetime_utc
        if pd.notna(row['datetime_utc']) and row['datetime_utc'] != '':
            try:
                return pd.to_datetime(row['datetime_utc'])
            except:
                return pd.to_datetime('1900-01-01')  # Far past date for missing values
        return pd.to_datetime('1900-01-01')
    
    # Combine start_date and start_time if available
    try:
        if pd.notna(row['start_time']) and row['start_time'] != '':
            datetime_str = f"{row['start_date']} {row['start_time']}"
            return pd.to_datetime(datetime_str)
        else:
            return pd.to_datetime(row['start_date'])
    except:
        return pd.to_datetime('1900-01-01')

# Add a temporary sorting column
df['sort_datetime'] = df.apply(parse_datetime_for_sorting, axis=1)

# Sort by the datetime (earliest first)
df = df.sort_values('sort_datetime')

# Remove the temporary sorting column
df = df.drop('sort_datetime', axis=1)

# Reset index after sorting
df = df.reset_index(drop=True)

# Save to CSV
filename = f"event_data_{datetime.now().strftime('%Y.%m.%d')}.csv"
df.to_csv(filename, index=False)
print(f"Saved {len(df)} events to {filename} (sorted by date)")