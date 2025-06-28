import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math

# Initialize lists
rows_date = []
rows_zone = []
rows_section = []
rows_rows = []
rows_quantity = []
rows_price = []

# Load CSV
#test
with open("C:/Users/zarak/Downloads/dataMariners/data4.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
        #test
    for row in csvreader:
        rows_date.append(row[0])
        rows_zone.append(row[1])
        rows_section.append(row[2])
        rows_rows.append(row[3])
        rows_quantity.append(row[4])
        rows_price.append(float(row[5]))

# Get unique zones
unique_zones = sorted(set(rows_zone))
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
zone_colors = {zone: colors[i % len(colors)] for i, zone in enumerate(unique_zones)}

# Row Quality Index
def compute_row_quality(zone, section, row):
    try:
        row_num = int(row)
    except ValueError:
        return 50

    score = max(0, 100 - row_num * 2)

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

# Group by zone
zone_grouped_data = defaultdict(list)
for date, zone, price, quality in zip(rows_date, rows_zone, rows_price, row_quality):
    zone_grouped_data[zone].append((date, price, quality))

# Subplot layout
num_zones = len(unique_zones)
cols = 3
rows = math.ceil(num_zones / cols)

fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharex=True)
axs = axs.flatten()

for i, zone in enumerate(unique_zones):
    ax = axs[i]
    data = zone_grouped_data[zone]
    if data:
        dates, prices, qualities = zip(*data)
        ax.scatter(dates, prices, s=np.array(qualities) / 2 + 10,
                   c=[zone_colors[zone]] * len(dates),
                   alpha=0.7, edgecolors='w', linewidths=0.5)
    ax.set_title(zone)
    ax.set_ylabel("Price ($)")
    ax.tick_params(axis='x', rotation=45)

# Turn off unused axes
for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])

fig.suptitle("Ticket Prices Over Time by Zone", fontsize=16)
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()