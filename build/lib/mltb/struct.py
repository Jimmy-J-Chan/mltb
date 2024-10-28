import pickle
import pandas as pd
import yaml # pip install PyYAML

class Struct:
    def __init__(self, obj="", df=False):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    setattr(self, k, Struct(v))
                else:
                    setattr(self, k, v)
        elif isinstance(obj, list):
            for k in obj:
                if df:
                    setattr(self, k, pd.DataFrame())
                else:
                    setattr(self, k, Struct())
        elif isinstance(obj, str) & obj.endswith('.yml'):
            with open(obj, 'r') as file:
                f = yaml.safe_load(file)

            for k, v in f.items():
                if isinstance(v, dict):
                    setattr(self, k, Struct(v))
                else:
                    setattr(self, k, v)
            pass

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, k, v):
        return setattr(self, k, v)

    @property
    def keys(self):
        return list(self.__dict__)

    def to_pickle(self, fn):
        with open(fn, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

