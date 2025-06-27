
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math
from datetime import datetime


# what needs to happen is that the filename of the SeatData CSV needs to be like
# something with the name and date. so like MarinersGuardians.06.17.2025
# in order to do this properly, i will need to use "selenium"
# once i got the file, i need to match it to the CSV file w/ the correct date and time

# Initialize lists for seat data
rows_date = []
rows_zone = []
rows_section = []
rows_rows = []
rows_quantity = []
rows_price = []

#days until event instead of date

with open("C:/Users/zarak/Downloads/TestData_Mariners/Seattle_Mariners_at_Minnesota_Twins_2025-06-25.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    
    for row in csvreader:
        rows_date.append(row[0])
        rows_zone.append(row[1])
        rows_section.append(row[2])
        rows_rows.append(row[3])
        rows_quantity.append(row[4])
        rows_price.append(float(row[5]))

# === STEP 2: Extract matching event metadata ===
# Define the target game we're looking for
target_game = "Seattle_Mariners_at_Minnesota_Twins_2025-06-25"
target_date = "2025-06-25"  # Extract date from filename

# Initialize individual variables for the matching event
event_id = None
event_name = None
url = None
start_date = None
start_time = None
datetime_utc = None
timezone = None
ticket_status = None
public_sale_start = None
public_sale_end = None
presales = None
image = None
info = None
please_note = None
seatmap_url = None
accessibility_info = None
ticket_limit = None
venue_id = None
venue_name = None
city = None
state = None
country = None
venue_timezone = None
segment = None
genre = None
subGenre = None
type_val = None
location = None
name = None
dates = None
sales = None
priceRanges = None
promoter = None
promoters = None
seatmap = None
accessibility = None
ticketLimit = None
classifications = None
externalLinks = None

#real code
event_csv_path = f"C:/Users/zarak/OneDrive/Documents/GitHub/ticket_model/event_data_{datetime.now().strftime('%Y.%m.%d')}.csv"

#for testing my model
event_csv_path = f"C:/Users/zarak/OneDrive/Documents/GitHub/ticket_model/event_data_2025.06.24.csv"

# Find and extract the matching event data
with open(event_csv_path, 'r', encoding='utf-8') as file:
    csvreader = csv.DictReader(file)
    for row in csvreader:
        # Check if this row matches our target date or event name
        if (row["start_date"] == target_date or 
            target_date in row.get("start_date", "") or
            "Seattle_Mariners_at_Minnesota_Twins" in row.get("event_name", "")):
            # Store all the values for the matching event
            event_id = row["event_id"]
            event_name = row["event_name"]
            url = row["url"]
            start_date = row["start_date"]
            start_time = row["start_time"]
            datetime_utc = row["datetime_utc"]
            timezone = row["timezone"]
            ticket_status = row["ticket_status"]
            public_sale_start = row["public_sale_start"]
            public_sale_end = row["public_sale_end"]
            presales = row["presales"]
            image = row["image"]
            info = row["info"]
            please_note = row["please_note"]
            seatmap_url = row["seatmap_url"]
            accessibility_info = row["accessibility_info"]
            ticket_limit = row["ticket_limit"]
            venue_id = row["venue_id"]
            venue_name = row["venue_name"]
            city = row["city"]
            state = row["state"]
            country = row["country"]
            venue_timezone = row["venue_timezone"]
            segment = row["segment"]
            genre = row["genre"]
            subGenre = row["subGenre"]
            type_val = row["type"]  # renamed to avoid conflict with Python's type()
            location = row["location"]
            name = row["name"]
            dates = row["dates"]
            sales = row["sales"]
            priceRanges = row["priceRanges"]
            promoter = row["promoter"]
            promoters = row["promoters"]
            seatmap = row["seatmap"]
            accessibility = row["accessibility"]
            ticketLimit = row["ticketLimit"]
            classifications = row["classifications"]
            externalLinks = row["externalLinks"]
            
            break  # Stop after finding the first match

# Print the extracted values to verify
print("Found matching event:")
print("event_name:", event_name)
print("start_date:", start_date)
print("start_time:", start_time)
print("datetime_utc:", datetime_utc)
print("venue_name:", venue_name)
print("city:", city)
print("state:", state)