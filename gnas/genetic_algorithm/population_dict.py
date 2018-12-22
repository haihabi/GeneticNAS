from collections import OrderedDict
from operator import itemgetter
import sys


class PopulationDict(object):
    def __init__(self, values_dict: OrderedDict = OrderedDict({}), index_dict: OrderedDict = OrderedDict({}),
                 current_index=0):
        self.values_dict = values_dict
        self.index_dict = index_dict
        self.i = current_index

    def __len__(self):
        return len(self.values_dict)

    def __str__(self):
        return str(self.values_dict)

    def items(self):
        return self.values_dict.items()

    def values(self):
        return self.values_dict.values()

    def keys(self):
        return self.values_dict.keys()

    def update(self, input_dict: dict):
        self.values_dict.update(input_dict)
        for k, v in input_dict.items():
            self.index_dict.update({k: self.i})
        self.i += 1

    def filter_top_n(self, n=sys.maxsize):
        values_dict = OrderedDict({})
        index_dict = OrderedDict({})
        for i, (key, value) in enumerate(sorted(self.values_dict.items(), key=itemgetter(1), reverse=True)):
            if i < n:
                values_dict.update({key: value})
                index_dict.update({key: self.index_dict.get(key)})
        return PopulationDict(values_dict, index_dict, self.i)

    def filter_last_n(self, n=sys.maxsize):
        values_dict = OrderedDict({})
        index_dict = OrderedDict({})
        for i, (key, index) in enumerate(sorted(self.index_dict.items(), key=itemgetter(1), reverse=True)):
            if i < n:
                values_dict.update({key: self.values_dict.get(key)})
                index_dict.update({key: index})
        return PopulationDict(values_dict, index_dict, self.i)

    def merge(self, other):
        values_dict = self.values_dict.copy()
        index_dict = self.index_dict.copy()
        values_dict.update(other.values_dict)
        index_dict.update(other.index_dict)
        return PopulationDict(values_dict, index_dict, self.i)
