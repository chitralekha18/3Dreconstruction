__author__ = 'abhinavkashyap'
import pickle

def pickledump(obj, file):
    """

    :param obj: object that has to be pickled
    :param file: file is an open file object for writing
    :return:
    """
    pickle.dump(obj, file)
