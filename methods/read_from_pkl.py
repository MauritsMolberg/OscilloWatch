import pickle


def read_from_pkl(file_path="results.pkl"):
    try:
        with open(file_path, 'rb') as file:
            loaded_segment_results = []
            while True:
                loaded_segment = pickle.load(file)
                loaded_segment_results.append(loaded_segment)
    except EOFError:
        pass  # End of file reached

    return loaded_segment_results
