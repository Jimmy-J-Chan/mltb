import pickle


def to_pickle(fn, obj):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(fn):
    with open(fn, 'rb') as f:
        obj = pickle.load(f)
    return obj
