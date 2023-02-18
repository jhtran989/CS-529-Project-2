import random

import pandas
from pandas import DataFrame

from Utilities import CLASS_NAME


class DataParameters:
    # def __init__(self, attribute_dict: dict = None, class_instance_list=None):
    #     self.attribute_dict = attribute_dict
    #     self.class_instance_list = class_instance_list

    def __init__(self, data_df: DataFrame, output_df: DataFrame, attribute_list: list):
        self.data_df = data_df
        self.output_df = output_df
        self.attribute_list = attribute_list

        # set extra members
        self.attribute_dict = self.get_attribute_dict(data_df, attribute_list)
        self.class_instance_list = self.get_class_instance_list(output_df)

    @staticmethod
    def get_attribute_dict(data_df, attribute_list):
        attribute_dict = {}

        for attribute in attribute_list:
            attribute_dict[attribute] = DataParameters.get_labels(data_df, attribute)

        return attribute_dict

    @staticmethod
    def get_class_instance_list(output_df):
        return pandas.unique(output_df[CLASS_NAME])

    @staticmethod
    def get_labels(data_df, attribute):
        return pandas.unique(data_df[attribute])

    def get_random_attributes(self, n):
        """
        Only choose an interval of attributes to look at

        Uniformly over interval [1, n] where n is the number of attributes left to look at in the reduced training
        data (current node)
        :return:
        """

        # need to subtract 1
        rand_num = random.randint(1, n - 1)

        # TODO: need to remove the "class" attribute from the randomization...
        attribute_names = list(self.attribute_dict.keys())
        attribute_names.remove(CLASS_NAME)

        # TypeError: Population must be a sequence.  For dicts or sets, use sorted(d).
        return random.sample(attribute_names, k=rand_num)

