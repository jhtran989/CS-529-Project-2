from __future__ import annotations
from typing import TYPE_CHECKING

import pandas

if TYPE_CHECKING:
    from tree.Tree import Tree, Node

from abc import abstractmethod
from enum import Enum, auto
from math import log2, pow

from Utilities import CLASS_NAME
from tree.TreeUtilities import get_class_instance_partition_dict, get_class_instance_partition_prop_dict


class InformationGainEnum(Enum):
    ENTROPY = auto()
    GINI_INDEX = auto()
    MISCLASSIFICATION_ERROR = auto()


INFORMATION_GAIN_DEBUG = True


def InformationGainFactory(information_gain_method: InformationGainEnum, tree: Tree, node: Node):
    information_gain_constructors_dict = {
        InformationGainEnum.ENTROPY: Entropy,
        InformationGainEnum.GINI_INDEX: GiniIndex,
        InformationGainEnum.MISCLASSIFICATION_ERROR: MisclassificationError,
    }

    return information_gain_constructors_dict[information_gain_method](information_gain_method, tree, node)


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
        current_training_data = node.current_training_data_df

        # get random attributes
        num_data_entries, num_attributes = current_training_data.shape
        random_attribute_list = data_parameters.get_random_attributes(num_attributes)

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

            if INFORMATION_GAIN_DEBUG:
                print(f"-------------------------------------------------------------------")
                print(f"Attribute: {attribute}")

            for attribute_value in attribute_instances:
                # FIXME: value_counts() can use proportion instead of raw count...
                attribute_value_count = current_training_data[attribute].value_counts()[attribute_value]
                attribute_value_prop = attribute_value_count / num_data_entries

                class_instances_attribute_value_df = \
                    current_training_data[current_training_data[attribute].isin([attribute_value])]

                class_instances_attribute_value_dict = \
                    get_class_instance_partition_dict(data_parameters,
                                                      class_instances_attribute_value_df)

                if INFORMATION_GAIN_DEBUG:
                    print(f"---------")

                # SOLVED: entropy values are greater than 1... -> used += instead of -=
                measure_attribute_value_weighted = attribute_value_prop * \
                                               self.calculate_measure_total(
                                                   class_instances_attribute_value_dict)
                measure_attribute[attribute] -= measure_attribute_value_weighted


                if INFORMATION_GAIN_DEBUG:
                    print(f"attribute value: {attribute_value}")
                    print(f"attribute value proportion: {attribute_value_prop}")
                    print(f"partial measure (weighted): {measure_attribute_value_weighted}")
                    # print(f"attribute proportion (value_counts): "
                    #       f"{current_training_data[attribute].value_counts(normalize=True)[attribute_value]}")
                    print(f"current data: \n{class_instances_attribute_value_df[[CLASS_NAME, attribute]]}")

            if INFORMATION_GAIN_DEBUG:
                print(f"---------")
                print(f"final measure: {measure_attribute[attribute]}")

        if INFORMATION_GAIN_DEBUG:
            print(f"-------------------------------------------------------------------")
            print(f"class instances: {node.class_instance_partition_dict}")
            print(f"measure parent: {measure_parent}")
            print(f"measure attribute: {measure_attribute}")

        return max(measure_attribute, key=measure_attribute.get)


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

