import csv

def write_data_to_csv(file_path, list_of_dicts, additional_values):
    # Extracting headers from the first dictionary in the list
    headers = list(list_of_dicts[0].keys())

    with open(file_path, 'w', newline='') as csv_file:
        # Creating a CSV writer object with semicolon as the delimiter
        csv_writer = csv.writer(csv_file, delimiter=';')

        # Writing the header row
        csv_writer.writerow(['Additional'] + headers)

        # Writing the values for each dictionary in the list along with additional values
        for additional_value, data_dict in zip(additional_values, list_of_dicts):
            # Writing each value separately
            csv_writer.writerow([additional_value] + [data_dict[header] for header in headers])

# Example usage:
data = [
    {'Name': 'John', 'Age': 30, 'City': 'New York'},
    {'Name': 'Alice', 'Age': 25, 'City': 'San Francisco'},
    {'Name': 'Bob', 'Age': 35, 'City': 'Los Angeles'}
]

additional_values = ['Value1', 'Value2', 'Value3']

write_data_to_csv('output.csv', data, additional_values)
