import os


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
