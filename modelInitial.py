import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Initialize lists
rows_date = []
rows_zone = []
rows_section = []
rows_rows = []
rows_quantity = []
rows_price = []

# Load CSV
with open("C:/Users/zarak/Downloads/DataSets/data4.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    
    for row in csvreader:
        rows_date.append(row[0])
        rows_zone.append(row[1])
        rows_section.append(row[2])
        rows_rows.append(row[3])
        rows_quantity.append(row[4])
        rows_price.append(float(row[5]))  # Ensure prices are floats

# Get unique zones
unique_zones = sorted(set(rows_zone))
zone_colors = {zone: plt.cm.tab20(i % 20) for i, zone in enumerate(unique_zones)}

# Optional: Row Quality Index
def compute_row_quality(zone, section, row):
    try:
        row_num = int(row)
    except ValueError:
        return 50  # Default score

    score = max(0, 100 - row_num * 2)

    # Simple boost/debuff logic
    if "Diamond" in zone:
        score += 15
    elif "Field" in zone:
        score += 10
    elif "Upper" in zone:
        score -= 10
    elif "Bleachers" in zone:
        score -= 15

    return np.clip(score, 0, 100)

row_quality = [compute_row_quality(z, s, r) for z, s, r in zip(rows_zone, rows_section, rows_rows)]

# Group data by zone for separate plotting
zone_grouped_data = defaultdict(list)

for date, zone, price, quality in zip(rows_date, rows_zone, rows_price, row_quality):
    zone_grouped_data[zone].append((date, price, quality))

# Plot
plt.figure(figsize=(14, 7))

for zone in unique_zones:
    data = zone_grouped_data[zone]
    if not data:
        continue
    dates, prices, qualities = zip(*data)
    plt.scatter(dates, prices, s=np.array(qualities) / 2 + 10,  # point size based on quality
                c=[zone_colors[zone]] * len(dates),
                label=zone, alpha=0.7, edgecolors='w', linewidths=0.5)

plt.xlabel("Date/Time")
plt.ylabel("Price ($)")
plt.title("Ticket Prices Over Time by Zone (Color & Size Encoded)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0), title="Zone", fontsize="small")
plt.tight_layout()
plt.show()