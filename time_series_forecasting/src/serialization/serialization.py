import pickle


def save_object(obj, output_file):
    with open(output_file, 'wb') as file:
        pickle.dump(obj, file)
    print("Saved object to file: {}".format(output_file))


def load_object(file):
    return pickle.load(open(file, "rb"))


def print_msg(msg, output_file_path):
    print(msg)
    if output_file_path != '':
        with open(output_file_path, 'a') as file:
            file.write(msg)
