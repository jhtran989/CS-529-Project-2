from enum import Enum, auto
import pandas

"""
Attribute names are generated from parsing the data (as a list)

"""

class Attribute:
    def __init__(self, attribute, labels):
        self.attribute = attribute
        self.labels = labels

    @staticmethod
    def get_labels(data_df, attribute):
        return pandas.unique(data_df[attribute])

    @staticmethod
    def get_attribute_dict(data_df, attribute_list):
        attribute_dict = {}

        for attribute in attribute_list:
            attribute_dict[attribute] = Attribute.get_labels(data_df, attribute)

        return attribute_dict


class ClassificationClass:
    def __init__(self, raw_class, instance_class):
        # TODO
        NotImplementedError

    @staticmethod
    def get_class_instance_list(output_df):
        return pandas.unique(output_df)


class Parameters:
    def __init__(self, attribute_dict, class_instance_list):
        self.attribute_dict = attribute_dict
        self.class_instance_list = class_instance_list

    def set_attribute_dict(self, data_df, attribute_list):
        attribute_dict = {}

        for attribute in attribute_list:
            attribute_dict[attribute] = Attribute.get_labels(data_df, attribute)

        self.attribute_dict = attribute_dict

    def set_class_instance_list(self, output_df):
        self.class_instance_list = pandas.unique(output_df)


class HyperParameters:
    def __init__(self):
        self.class_instance_cutoff_ratio = None
        self.chi_square_alpha = None
        self.information_gain = None


class InformationGainEnum(Enum):
    ENTROPY = auto()
    GINI_INDEX = auto()
    MISCLASSIFICATION_ERROR = auto()
