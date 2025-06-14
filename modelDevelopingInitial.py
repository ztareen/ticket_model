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

