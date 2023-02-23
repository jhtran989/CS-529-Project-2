from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tree.Tree import Tree, Node

import pandas as pd

from utilities.DebugFlags import INFORMATION_GAIN_DEBUG, INFORMATION_GAIN_PRINT

from abc import abstractmethod
from math import log2, pow
import numpy as np

from utilities.ParseUtilities import CLASS_NAME, MISSING_DATA_VALUE
from utilities.TreeUtilities import get_df_row_count, \
    get_class_instance_partition_dict, \
    get_class_instance_partition_prop_dict
from utilities.InformationGainUtilities import get_normalized_prob, get_uniform_prob, InformationGainEnum


def InformationGainFactory(information_gain_method: InformationGainEnum, tree: Tree, node: Node):
    information_gain_constructors_dict = {
        InformationGainEnum.ENTROPY: Entropy,
        InformationGainEnum.GINI_INDEX: GiniIndex,
        InformationGainEnum.MISCLASSIFICATION_ERROR: MisclassificationError,
    }

    return information_gain_constructors_dict[information_gain_method](information_gain_method, tree, node)


# @auto_str
class InformationGain:
    def __init__(self,
                 information_gain_method: InformationGainEnum,
                 tree: Tree,
                 node: Node):
        self.information_gain_method = information_gain_method
        self.tree = tree
        self.node = node

    def update_node(self, node: Node):
        self.node = node

    @abstractmethod
    def calculate_measure_total(self, class_instances_dict):
        pass

    @abstractmethod
    def calculate_measure_partial_p(self, prop_key_value):
        """
        Entropy: -(p log2 p)
        Gini Index: 1 - p^2
        Misclassification Error (ME): min{p, 1 - p}

        :param prop_key_value:
        :return:
        """

        pass

    def find_split(self):
        """
        Find the split at a given node (subset of data looked at) in a tree (global attributes and attribute values)

        :return:
        """

        tree = self.tree
        node = self.node

        data_parameters = tree.data_parameters
        attribute_dict = tree.data_parameters.attribute_dict

        # FIXME: need to create a copy so handling of missing data doesn't propagate to other levels...
        current_training_data = node.current_training_data_df[:]

        # get random attributes
        num_data_entries, num_attributes = current_training_data.shape

        attribute_visited_list = node.attribute_visited_list.copy()

        # SOLVED: need to remove attributes already visited from random list...
        # FIXME: update with hyper parameters to limit max number of attributes checked
        # random_attribute_list = data_parameters.get_random_attributes(attribute_visited_list, num_attributes)
        max_num_attributes_check = self.tree.hyper_parameters.max_num_attributes_check
        random_attribute_list = \
            data_parameters.get_random_attributes_max_num(attribute_visited_list, max_num_attributes_check)

        if INFORMATION_GAIN_DEBUG:
            print(f"shape of data: {current_training_data.shape}")

        # SOLVED: create DataFrame from the dict... -> just use a dict with a custom function to sum values (find
        #  proportions)

        class_instances_parent_dict = node.class_instance_partition_dict

        measure_parent = self.calculate_measure_total(class_instances_parent_dict)

        # SOLVED: store list of entropy values for each attribute -> use a dict
        measure_attribute = {}

        for attribute in random_attribute_list:
            attribute_instances = attribute_dict[attribute]

            measure_attribute[attribute] = measure_parent

            # FIXME: reorganized code to get the counts of each attribute instance initially
            attribute_instances_count_dict = {}
            for attribute_value in attribute_instances:
                attribute_value_count = get_df_row_count(current_training_data, attribute, attribute_value)
                attribute_instances_count_dict[attribute_value] = attribute_value_count

            total_non_missing_data_entries = sum(attribute_instances_count_dict.values())

            if INFORMATION_GAIN_DEBUG:
                print(f"-------------------------------------------------------------------")
                print(f"Attribute: {attribute}")
                print(f"attribute instances: {attribute_dict[attribute]}")
                print(f"focused data:")
                print(f"{current_training_data[[CLASS_NAME, attribute]]}")
                print(f"attribute instances count: {attribute_instances_count_dict}")
                print(f"num rows: {len(current_training_data[attribute])}")
                print(f"num non-missing rows: {total_non_missing_data_entries}")

            # TODO: need to handle MISSING DATA (at each split)
            # uses the same proportion as the non-missing data to fill in the missing data

            # if current_training_data[attribute].eq(MISSING_DATA_VALUE).any():
            if total_non_missing_data_entries < num_data_entries:
                current_training_data[attribute] = \
                    current_training_data[attribute].replace(MISSING_DATA_VALUE, np.NaN)

                # TODO: what if all the data is missing...
                if total_non_missing_data_entries == 0:
                    uniform_prob = get_uniform_prob(attribute_instances_count_dict)

                    current_training_data[attribute] = \
                        current_training_data[attribute].fillna(pd.Series(np.random.choice(
                            attribute_instances,
                            p=uniform_prob, size=len(current_training_data))))
                else:
                    non_missing_normalized_prob = get_normalized_prob(attribute_instances_count_dict,
                                                                      total_non_missing_data_entries)

                    current_training_data[attribute] = \
                        current_training_data[attribute].fillna(pd.Series(np.random.choice(
                            attribute_instances,
                            p=non_missing_normalized_prob, size=len(current_training_data))))

                if INFORMATION_GAIN_DEBUG:
                    print("MISSING DATA")
                    print("processed data:")
                    print(current_training_data[[CLASS_NAME, attribute]])

            for attribute_value in attribute_instances:
                # SOLVED: value_counts() can use proportion instead of raw count...
                # handle exception from value_counts()
                # attribute_value_count = current_training_data[attribute].value_counts()[attribute_value]
                attribute_value_count = get_df_row_count(current_training_data, attribute, attribute_value)
                attribute_value_prop = attribute_value_count / num_data_entries

                class_instances_attribute_value_df = \
                    current_training_data[current_training_data[attribute].isin([attribute_value])]

                class_instances_attribute_value_dict = \
                    get_class_instance_partition_dict(data_parameters,
                                                      class_instances_attribute_value_df)

                if INFORMATION_GAIN_DEBUG:
                    print(f"---------")
                    print(f"attribute value: {attribute_value}")
                    print(f"attribute value proportion: {attribute_value_prop}")
                    print(f"current data: \n{class_instances_attribute_value_df[[CLASS_NAME, attribute]]}")
                    print(f"class instance dict: {class_instances_attribute_value_dict}")

                # SOLVED: entropy values are greater than 1... -> used += instead of -=
                measure_attribute_value_weighted = attribute_value_prop * \
                                                   self.calculate_measure_total(
                                                       class_instances_attribute_value_dict)
                measure_attribute[attribute] -= measure_attribute_value_weighted

                if INFORMATION_GAIN_DEBUG:
                    print(f"partial measure (weighted): {measure_attribute_value_weighted}")
                    # print(f"attribute proportion (value_counts): "
                    #       f"{current_training_data[attribute].value_counts(normalize=True)[attribute_value]}")

            if INFORMATION_GAIN_DEBUG:
                print(f"---------")
                print(f"final measure: {measure_attribute[attribute]}")

        chosen_attribute = max(measure_attribute, key=measure_attribute.get)
        attribute_instances = attribute_dict[chosen_attribute]

        attribute_instances_count_dict = {}
        class_partition_attribute_values_dict = {}
        for attribute_value in attribute_instances:
            attribute_value_count = get_df_row_count(current_training_data, chosen_attribute, attribute_value)
            attribute_instances_count_dict[attribute_value] = attribute_value_count

            class_instances_attribute_value_df = \
                current_training_data[current_training_data[chosen_attribute].isin([attribute_value])]

            class_instances_attribute_value_dict = \
                get_class_instance_partition_dict(data_parameters,
                                                  class_instances_attribute_value_df)

            class_partition_attribute_values_dict[attribute_value] = class_instances_attribute_value_dict

        if INFORMATION_GAIN_PRINT:
            print(f"-------------------------------------------------------------------------------------------")
            print(f"class instances: {node.class_instance_partition_dict}")
            print(f"measure parent: {measure_parent}")
            print(f"measure attribute: {measure_attribute}")
            print(f"-------------------------------------------------------------------")
            print(f"chosen attribute: {chosen_attribute}")
            print(f"attribute values dict: {attribute_instances_count_dict}")
            print(f"class partition: {class_partition_attribute_values_dict}")
            print(f"-------------------------------------------------------------------------------------------")

        return chosen_attribute


# @auto_str
class Entropy(InformationGain):
    def __init__(self,
                 information_gain_method: InformationGainEnum,
                 tree: Tree,
                 node: Node):
        InformationGain.__init__(self, information_gain_method, tree, node)

    def calculate_measure_partial_p(self, prop_key_value):
        prop = prop_key_value[1]

        if INFORMATION_GAIN_DEBUG:
            print(f"class instance: {prop_key_value[0]}, prop: {prop}")

        if prop == 0:
            return 0

        return -(prop * log2(prop))

    def calculate_measure_total(self, class_instances_dict):
        return sum(map(
            lambda x: self.calculate_measure_partial_p(x),
            get_class_instance_partition_prop_dict(class_instances_dict).items()))

    def find_entropy(self, num_class_instances, total_instances):
        prop = num_class_instances / total_instances

        # special case where proportion is 0
        if prop == 0:
            return 0

        return -(prop * log2(prop))


# @auto_str
class GiniIndex(InformationGain):
    def __init__(self,
                 information_gain_method: InformationGainEnum,
                 tree: Tree,
                 node: Node):
        InformationGain.__init__(self, information_gain_method, tree, node)

    def calculate_measure_partial_p(self, prop_key_value):
        prop = prop_key_value[1]

        if INFORMATION_GAIN_DEBUG:
            print(f"class instance: {prop_key_value[0]}, prop: {prop}")

        return pow(prop, 2)

    def calculate_measure_total(self, class_instances_dict):
        return 1 - sum(map(
            lambda x: self.calculate_measure_partial_p(x),
            get_class_instance_partition_prop_dict(class_instances_dict).items()))


# @auto_str
class MisclassificationError(InformationGain):
    def __init__(self,
                 information_gain_method: InformationGainEnum,
                 tree: Tree,
                 node: Node):
        InformationGain.__init__(self, information_gain_method, tree, node)

    def calculate_measure_partial_p(self, prop_key_value):
        """
        Just return the prop

        :param prop_key_value:
        :return:
        """
        prop = prop_key_value[1]

        if INFORMATION_GAIN_DEBUG:
            print(f"class instance: {prop_key_value[0]}, prop: {prop}")

        return prop

    def calculate_measure_total(self, class_instances_dict):
        return 1 - max(map(
            lambda x: self.calculate_measure_partial_p(x),
            get_class_instance_partition_prop_dict(class_instances_dict).items()))
