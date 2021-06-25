import pickle


def save_object(obj, output_file):
    with open(output_file, 'wb') as file:
        pickle.dump(obj, file)
    print("Saved object to file: {}".format(output_file))


def load_object(file):
    return pickle.load(open(file, "rb"))
