import pickle
import itertools

def read_data_iterator(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        for i in itertools.count():
            try:
                yield pickle.load(f)
            except EOFError:
                raise StopIteration
                
            
def story_read_data_iterator(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        return pickle.load(f)