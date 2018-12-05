import os
import pickle


class ResultAppender(object):
    def __init__(self):
        self.result_dict = dict()

    def add_epoch_result(self, result_name: str, result_var: float):
        if self.result_dict.get(result_name) is None:
            self.result_dict.update({result_name: [result_var]})
        else:
            self.result_dict.get(result_name).append(result_var)

    def add_result(self, result_name: str, result_array):
        self.result_dict.update({result_name: result_array})

    def save_result(self, input_path):
        pickle.dump(self, open(os.path.join(input_path, 'ga_result.pickle'), "wb"))

    @staticmethod
    def load_result(input_path):
        return pickle.load(open(os.path.join(input_path, 'ga_result.pickle'), "rb"))
