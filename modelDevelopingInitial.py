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

# Initialize lists
rows_date = []
rows_zone = []
rows_section = []
rows_rows = []
rows_quantity = []
rows_price = []

#days until event instead of date

with open("C:/Users/zarak/Downloads/dataMariners/data4.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    
    for row in csvreader:
        rows_date.append(row[0])
        rows_zone.append(row[1])
        rows_section.append(row[2])
        rows_rows.append(row[3])
        rows_quantity.append(row[4])
        rows_price.append(float(row[5]))

# === STEP 2: Load event metadata into arrays ===
# Initialize all arrays for event metadata
event_id_list = []
event_name_list = []
url_list = []
start_date_list = []
start_time_list = []
datetime_utc_list = []
timezone_list = []
ticket_status_list = []
public_sale_start_list = []
public_sale_end_list = []
presales_list = []
image_list = []
info_list = []
please_note_list = []
seatmap_url_list = []
accessibility_info_list = []
ticket_limit_list = []
venue_id_list = []
venue_name_list = []
city_list = []
state_list = []
country_list = []
venue_timezone_list = []
segment_list = []
genre_list = []
subGenre_list = []
type_list = []
location_list = []
name_list = []
dates_list = []
sales_list = []
priceRanges_list = []
promoter_list = []
promoters_list = []
seatmap_list = []
accessibility_list = []
ticketLimit_list = []
classifications_list = []
externalLinks_list = []

event_csv_path = f"C:\\Users\\zarak\\OneDrive\\Documents\\GitHub\\ticket_model\\event_data_{datetime.now().strftime('%Y.%m.%d')}.csv"

with open(event_csv_path, 'r', encoding='utf-8') as file:
    csvreader = csv.DictReader(file)
    for row in csvreader:
        event_id_list.append(row["event_id"])
        event_name_list.append(row["event_name"])
        url_list.append(row["url"])
        start_date_list.append(row["start_date"])
        start_time_list.append(row["start_time"])
        datetime_utc_list.append(row["datetime_utc"])
        timezone_list.append(row["timezone"])
        ticket_status_list.append(row["ticket_status"])
        public_sale_start_list.append(row["public_sale_start"])
        public_sale_end_list.append(row["public_sale_end"])
        presales_list.append(row["presales"])
        image_list.append(row["image"])
        info_list.append(row["info"])
        please_note_list.append(row["please_note"])
        seatmap_url_list.append(row["seatmap_url"])
        accessibility_info_list.append(row["accessibility_info"])
        ticket_limit_list.append(row["ticket_limit"])
        venue_id_list.append(row["venue_id"])
        venue_name_list.append(row["venue_name"])
        city_list.append(row["city"])
        state_list.append(row["state"])
        country_list.append(row["country"])
        venue_timezone_list.append(row["venue_timezone"])
        segment_list.append(row["segment"])
        genre_list.append(row["genre"])
        subGenre_list.append(row["subGenre"])
        type_list.append(row["type"])
        location_list.append(row["location"])
        name_list.append(row["name"])
        dates_list.append(row["dates"])
        sales_list.append(row["sales"])
        priceRanges_list.append(row["priceRanges"])
        promoter_list.append(row["promoter"])
        promoters_list.append(row["promoters"])
        seatmap_list.append(row["seatmap"])
        accessibility_list.append(row["accessibility"])
        ticketLimit_list.append(row["ticketLimit"])
        classifications_list.append(row["classifications"])
        externalLinks_list.append(row["externalLinks"])




print("event_name_list:", event_name_list)
print("start_date_list:", start_date_list)
print("start_time_list:", start_time_list)
print("datetime_utc_list:", datetime_utc_list)