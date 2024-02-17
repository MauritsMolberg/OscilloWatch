import pickle

class TestClass:
    def __init__(self, result):
        self.result = result


def clear_file(file_path="test.pkl"):
    with open(file_path, "wb") as file:
        pass

# Function to add a new segment result to the file
def add_segment_result(obj, file_path='test.pkl'):
    with open(file_path, 'ab') as file:
        pickle.dump(obj, file)

# Example: Add new TestClass instances to the file as they are obtained
segment1 = TestClass("aaaæ")
segment2 = TestClass(3)

clear_file()
add_segment_result(segment1)
add_segment_result(segment2)


# To read the stored results from the file
def read_segment_results(file_path='test.pkl'):
    try:
        with open(file_path, 'rb') as file:
            loaded_segment_results = []
            while True:
                loaded_segment = pickle.load(file)
                loaded_segment_results.append(loaded_segment)
    except EOFError:
        pass

    return loaded_segment_results


# Example: Read the stored results from the file
loaded_results = read_segment_results()
print([res.result for res in loaded_results])


