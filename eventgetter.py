import requests
import json

from ticketInfoGetter import ticketInfoGetter
#from seatingInfoGetter import seatingInfoGetter

event_IDs_final = []

url = "https://api.seatgeek.com/2/events?client_id=NTAzNDk2MTZ8MTc0ODAyMzY5MS41NzUwNDY&q=mlb"
#url_second = "https://api.seatgeek.com/2/events/section_info/:eventId"

response = requests.get(url)
#secondRepsonse = requests.get(url_second)

# Parse JSON response into a Python dict
data = response.json()
#secondData = secondRepsonse.json()

# Now your code can access data["events"]
events = data["events"]
#events_seating = data["seat"]

for event in events:
    currentID = ticketInfoGetter(event)
    event_IDs_final.append(currentID)


print(event_IDs_final)

#for seat in seats:
 #   seatingInfoGetter(event)
