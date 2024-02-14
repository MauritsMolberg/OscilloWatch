import csv

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

with open("test.csv", 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=";")

    row = [i + 1]
    for header in headers:
        row.append(data_dict[header])
    csv_writer.writerow(row)