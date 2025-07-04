import requests
import json

from ticketInfoGetter import ticketInfoGetter
from seatingInfoGetter import seatingInfoGetter

event_IDs_final = []
seatingURLs = []

# Get first page of MLB-related events
url = "https://api.seatgeek.com/2/events?performers.slug=seattle-mariners&listing_count.gt=0&client_id=NTAzNDk2MTZ8MTc0ODAyMzY5MS41NzUwNDY"

response = requests.get(url)
data = response.json()
events = data["events"]

i = 0

for event in events:
    currentID = ticketInfoGetter(event)
    event_IDs_final.append(currentID)

    full_url = f"https://api.seatgeek.com/2/events/section_info/{currentID}?client_id=NTAzNDk2MTZ8MTc0ODAyMzY5MS41NzUwNDY"
    seatingURLs.append(full_url)
    
    #print(f"The URL for this would be {full_url}")
    response2 = requests.get(full_url)
    data2 = response2.json()

    sections = data2.get("sections", {})

    for section_name, rows in sections.items():
        seatingInfoGetter(section_name, rows)

    # Extract lowest price from the event data if it exists
    lowest_price = data2.get("lowest_price", 0)
    if lowest_price is None:
        lowest_price = 0

    if lowest_price <= 0:
        i = 0  # error or no price info
    elif lowest_price < 30:
        i = 3
    elif lowest_price < 60:
        i = 10
    elif lowest_price < 90:
        i = 20
    elif lowest_price < 150:
        i = 30
    elif lowest_price < 250:
        i = 40
    else:
        i = 50  # For prices 250 or above
    print(i)

    i += 1

#print(event_IDs_final)

print(i)