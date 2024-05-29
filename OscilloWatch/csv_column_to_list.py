#################################################################################
#     OscilloWatch, early warning application for low-frequency oscillations
#     Copyright (C) 2024  Maurits Sørensen Molberg
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#################################################################################

import csv


def csv_column_to_list(file_path, column_index, delimiter=","):
    """
    Simple function for reading from a CSV file and putting all non-NaN elements in a list. The top row is assumed to
    contain headers and will not be included in the list.

    :param str file_path: File path to the CSV file that is to be read.
    :param int column_index: Index of the row that is to be read and put in a list.
    :param str delimiter: Symbol used as delimiter in the CSV file. Comma is the most common, but semicolon is typically
        used in countries where the comma is used for decimal numbers.
    :return: List containing non-NaN elements of the selected row from the CSV file.
    :rtype: list
    """
    values = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        headers = next(csv_reader)  # Skip the header row
        for row in csv_reader:
            if row[column_index].lower() != "nan" and row[column_index].lower() != "#num!" and len(row) > column_index:
                values.append(float(row[column_index]))
    return values
