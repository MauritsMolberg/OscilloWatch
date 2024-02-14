import csv
import datetime
from time import time

headers = [
            "Warning",
            "Freq. start",
            "Freq. stop",
            "Start time",
            "End time",
            "NZF",
            "Init. amp.",
            "Final amp.",
            "Init. amp. est.",
            "Decay rate",
            "Damping ratio",
            "Interp. frac.",
            "CV",
            "Note"
           ]

# Adds _new to file name if permission denied (when file is open in Excel, most likely)

with open("test.csv", 'a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=";")

    row = [1,2,"heisann", 5, 8.422, datetime.datetime.now().strftime("%Y.%m.%d, %H:%M:%S.%f")]
    csv_writer.writerow(row)

#print(datetime.datetime.now().strftime("%Y.%m.%d, %H:%M:%S.%f"))
timestamp = time()
dt_object = datetime.datetime.fromtimestamp(timestamp)
print(dt_object)