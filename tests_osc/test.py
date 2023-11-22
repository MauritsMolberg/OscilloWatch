import csv

def write_dicts_to_csv(file_path, list_of_dicts):
    # Extracting headers from the first dictionary in the list
    headers = list(list_of_dicts[0].keys())

    with open(file_path, 'w', newline='') as csv_file:
        # Creating a CSV writer object with semicolon as the delimiter
        csv_writer = csv.writer(csv_file, delimiter=';')

        # Writing the headers to the CSV file
        csv_writer.writerow(headers)

        # Writing the values for each dictionary in the list
        for data_dict in list_of_dicts:
            # Writing each value separately
            csv_writer.writerow([data_dict[header] for header in headers])

# Example usage:
data = [
    {'Name': 'John', 'Age': 30, 'City': 'New York'},
    {'Name': 'Alice', 'Age': 25, 'City': 'San Francisco'},
    {'Name': 'Bob', 'Age': 35, 'City': 'Los Angeles'}
]

write_dicts_to_csv('output.csv', data)
