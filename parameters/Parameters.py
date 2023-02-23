import random

import numpy as np
import pandas
from pandas import DataFrame

from utilities.ParseUtilities import CLASS_NAME, MISSING_DATA_VALUE
from utilities.PrintUtilities import auto_str


@auto_str
class DataParameters:
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
    def get_class_instance_list(output_df: DataFrame):
        # unique_attribute_instances_array = pandas.unique(output_df[CLASS_NAME])
        #
        # # need to remove instances that represent MISSING DATA
        # unique_attribute_instances_no_missing_array = np.delete(unique_attribute_instances_array,
        #                                              np.where(unique_attribute_instances_array ==
        #                                                       MISSING_DATA_VALUE))

        return DataParameters.get_labels(output_df, CLASS_NAME)

    @staticmethod
    def get_labels(data_df, attribute):
        # need to remove instances that represent MISSING DATA
        unique_labels_array = pandas.unique(data_df[attribute])
        unique_labels_no_missing_array = np.delete(unique_labels_array,
                                                                np.where(unique_labels_array ==
                                                                         MISSING_DATA_VALUE))

        # FIXME: just return a list instead of a numpy array
        return list(unique_labels_no_missing_array)

    def get_random_attributes_max_num(self, attribute_visited_list: list, max_num_attributes_check):
        """
        Same as below except the interval is now [1, min(max_num, num_remaining_attributes)]
        
        max_num is carried over from the hyper parameters
        
        :param attribute_visited_list: 
        :param max_num_attributes_check: 
        :return: 
        """

        # TODO: need to remove the "class" attribute from the randomization...
        attribute_names = list(self.attribute_dict.keys())
        attribute_names.remove(CLASS_NAME)

        # need to take the set difference so we don't look at the same attribute along the same path
        set_diff = set(attribute_names).symmetric_difference(set(attribute_visited_list))
        remaining_attributes_list = list(set_diff)

        num_remaining_attributes = len(remaining_attributes_list)

        # need to subtract 1 - INCLUSIVE
        # FIXME: need to add 1 so randint(1, 1 + 1) would return 1 for just one remaining attribute
        # FIXME: actually, don't need extra modification...
        num_attributes_check = min(num_remaining_attributes, max_num_attributes_check)
        rand_num = random.randint(1, num_attributes_check)

        # FIXME: the set_diff should NOT be empty...
        assert (num_remaining_attributes > 0)

        # TypeError: Population must be a sequence.  For dicts or sets, use sorted(d).
        return random.sample(remaining_attributes_list,
                             k=rand_num)

    def get_random_attributes(self, attribute_visited_list: list, n):
        """
        Only choose an interval of attributes to look at

        Uniformly over interval [1, n] where n is the number of attributes left to look at in the reduced training
        data (current node)
        :param n:
        :param attribute_visited_list:
        :return:
        """

        # TODO: need to remove the "class" attribute from the randomization...
        attribute_names = list(self.attribute_dict.keys())
        attribute_names.remove(CLASS_NAME)

        # need to take the set difference so we don't look at the same attribute along the same path
        set_diff = set(attribute_names).symmetric_difference(set(attribute_visited_list))
        remaining_attributes_list = list(set_diff)

        num_remaining_attributes = len(remaining_attributes_list)

        # need to subtract 1 - INCLUSIVE
        rand_num = random.randint(1, num_remaining_attributes - 1)

        # FIXME: the set_diff should NOT be empty...
        assert(num_remaining_attributes > 0)

        # TypeError: Population must be a sequence.  For dicts or sets, use sorted(d).
        return random.sample(remaining_attributes_list, k=rand_num)

