import requests
import json

from ticketInfoGetter import ticketInfoGetter
from seatingInfoGetter import seatingInfoGetter

#mlb_slugs = {
    #"arizona-diamondbacks", "atlanta-braves", "baltimore-orioles", "boston-red-sox",
    #"chicago-cubs", "chicago-white-sox", "cincinnati-reds", "cleveland-guardians",
    #"colorado-rockies", "detroit-tigers", "houston-astros", "kansas-city-royals",
    #"los-angeles-angels", "los-angeles-dodgers", "miami-marlins", "milwaukee-brewers",
    #"minnesota-twins", "new-york-mets", "new-york-yankees", "oakland-athletics",
    #"philadelphia-phillies", "pittsburgh-pirates", "san-diego-padres", "san-francisco-giants",
    #"seattle-mariners", "st-louis-cardinals", "tampa-bay-rays", "texas-rangers",
    #"toronto-blue-jays", "washington-nationals"
#}

event_IDs_final = []
seatingURLs = []

# Get first page of MLB-related events
url = "https://api.seatgeek.com/2/events?performers.slug=seattle-mariners&client_id=NTAzNDk2MTZ8MTc0ODAyMzY5MS41NzUwNDY"


response = requests.get(url)
#secondRepsonse = requests.get(url_second)

# Parse JSON response into a Python dict
data = response.json()
#secondData = secondRepsonse.json()

# Now your code can access data["events"]
events = data["events"]
#events_seating = data["seat"]

i = 0

for event in events:
    currentID = ticketInfoGetter(event)
    event_IDs_final.append(currentID)

    full_url = f"https://api.seatgeek.com/2/events/section_info/{currentID}?client_id=NTAzNDk2MTZ8MTc0ODAyMzY5MS41NzUwNDY"
    seatingURLs.append(full_url)


    print(f"The URL for this would be {full_url}")
    i = i + 1


#for section in sections:
   # seatingInfoGetter(event, currentID)

print(event_IDs_final)