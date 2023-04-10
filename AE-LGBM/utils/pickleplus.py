import pickle


def dump(data, filename: str):
    pickle.dump(data, open(filename, "wb"))


def dump_lst(lst_data, lst_filename):
    for data, filename in zip(lst_data, lst_filename):
        dump(data, filename)


def load(filename: str):
    return pickle.load(open(filename, "rb"))


def load_lst(lst_filename):
    data = []
    for filename in lst_filename:
        data.append(load(filename))

    return data
