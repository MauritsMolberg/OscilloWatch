import pickle


def read_from_pkl(file_path, start_index=None, end_index=None):
    try:
        with open(file_path, 'rb') as file:
            loaded_objects = []
            current_index = 0

            while True:
                try:
                    loaded_object = pickle.load(file)
                except EOFError:
                    break  # End of file reached

                if (start_index is None or current_index >= start_index) and \
                   (end_index is None or current_index <= end_index):
                    loaded_objects.append(loaded_object)

                current_index += 1

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error occurred while reading from '{file_path}': {e}")
        return []

    return loaded_objects
