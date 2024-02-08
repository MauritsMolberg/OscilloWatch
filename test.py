import pickle

class TestClass:
    def __init__(self, result):
        self.result = result

# Function to add a new segment result to the file
def add_segment_result(segment_result, file_path='segment_results.pkl'):
    with open(file_path, 'ab') as file:
        pickle.dump(segment_result, file)

# Example: Add new TestClass instances to the file as they are obtained
segment1 = TestClass("aaa")
add_segment_result(segment1)

segment2 = TestClass(3)
add_segment_result(segment2)

# ...


# To read the stored results from the file
def read_segment_results(file_path='segment_results.pkl'):
    try:
        with open(file_path, 'rb') as file:
            loaded_segment_results = []
            while True:
                loaded_segment = pickle.load(file)
                loaded_segment_results.append(loaded_segment)
    except EOFError:
        pass  # End of file reached

    return loaded_segment_results


# Example: Read the stored results from the file
loaded_results = read_segment_results()
for segment in loaded_results:
    print(segment.result)
