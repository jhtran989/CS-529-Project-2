import pandas
import random

from InformationGain import InformationGainEnum


class DataParameters:
    def __init__(self, attribute_dict: dict, class_instance_list):
        self.attribute_dict = attribute_dict
        self.class_instance_list = class_instance_list

    @staticmethod
    def get_labels(data_df, attribute):
        return pandas.unique(data_df[attribute])

    def set_attribute_dict(self, data_df, attribute_list):
        attribute_dict = {}

        for attribute in attribute_list:
            attribute_dict[attribute] = DataParameters.get_labels(data_df, attribute)

        self.attribute_dict = attribute_dict

    def set_class_instance_list(self, output_df):
        self.class_instance_list = pandas.unique(output_df)

    def get_random_attributes(self, n):
        """
        Only choose an interval of attributes to look at

        Uniformly over interval [1, n] where n is the number of attributes left to look at in the reduced training
        data (current node)
        :return:
        """

        rand_num = random.randint(1, n)

        return random.sample(self.attribute_dict.keys(), k=rand_num)

class HyperParameters:
    def __init__(self,
                 class_instance_cutoff_ratio,
                 chi_square_alpha,
                 information_gain: InformationGainEnum,
                 num_attributes_interval,
                 num_trees,
                 percent_training_validation):
        self.class_instance_cutoff_ratio = class_instance_cutoff_ratio
        self.chi_square_alpha = chi_square_alpha
        self.information_gain = information_gain
        self.num_attributes_interval = num_attributes_interval
        self.num_trees = num_trees
        self.percent_training_validation = percent_training_validation

