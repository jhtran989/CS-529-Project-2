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


