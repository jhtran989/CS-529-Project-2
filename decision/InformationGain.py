from __future__ import annotations
from typing import TYPE_CHECKING

import pandas

if TYPE_CHECKING:
    from tree.Tree import Tree, Node

from abc import abstractmethod
from enum import Enum, auto
from math import log2

from Utilities import CLASS_NAME
from tree.TreeUtilities import get_class_instance_partition_dict, get_class_instance_partition_prop_dict


class InformationGainEnum(Enum):
    ENTROPY = auto()
    GINI_INDEX = auto()
    MISCLASSIFICATION_ERROR = auto()


INFORMATION_GAIN_DEBUG = True


class InformationGain:
    def __init__(self,
                 information_gain_method: InformationGainEnum,
                 tree: Tree,
                 node: Node):
        self.information_gain_method = information_gain_method
        self.tree = tree
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

        # SOLVED: create DataFrame from the dict... -> just use a dict with a custom function to sum values
        # class_instances_parent = node.class_instance_partition_dict.values()
        # num_class_instances_parent = sum(class_instances_parent)

        # NUM_CLASS_NAME = "num_class"
        #
        # class_instances_parent = pandas.DataFrame(list(node.class_instance_partition_dict.items()),
        #                                           columns=[CLASS_NAME, NUM_CLASS_NAME])

        # entropy_parent is the entropy of the curren node - Entropy(S)
        # entropy_parent = -sum(map(lambda x: (x / num_instances_parent) * log2(x / num_instances_parent)))
        # entropy_parent = sum(map(lambda x: self.calculate_measure(x, num_class_instances_parent),
        #                          class_instances_parent))

        # entropy_parent = sum(map(
        #     lambda x: self.calculate_measure(pandas.Series(class_instances_parent)
        #                                      .value_counts(normalize=True)[x]),
        #     class_instances_parent))

        class_instances_parent_dict = node.class_instance_partition_dict

        # measure_parent = sum(map(
        #     lambda x: self.calculate_measure_partial_p(x),
        #     get_class_instance_partition_prop_dict(class_instances_parent_dict).items()))

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

                # measure_attribute[attribute] -= attribute_value_prop * \
                #                                sum(map(
                #                                    lambda x: self.calculate_measure_partial_p(x),
                #                                    get_class_instance_partition_prop_dict(
                #                                        class_instances_attribute_value_dict).items()))
                measure_attribute_value_weighted = attribute_value_prop * \
                                               self.calculate_measure_total(
                                                   class_instances_attribute_value_dict)
                measure_attribute[attribute] -= measure_attribute_value_weighted

                # class_instances_attribute_value = \
                #     get_class_instance_partition_dict(data_parameters,
                #                                       class_instances_attribute_value_df).values()
                # num_class_instances_attribute_value = sum(class_instances_attribute_value)

                # SOLVED: entropy values are greater than 1... -> used += instead of -=
                # entropy_attribute[attribute] += attribute_value_prop * \
                #                                 self.find_entropy(class_instances_attribute_value)
                # entropy_attribute[attribute] -= attribute_value_prop * \
                #                                 sum(map(lambda x:
                #                                         self.calculate_measure(x, num_class_instances_attribute_value),
                #                                         class_instances_attribute_value))
                # entropy_attribute[attribute] -= attribute_value_prop * \
                #                                 sum(map(lambda x:
                #                                         self.calculate_measure(
                #                                             pandas.Series(class_instances_attribute_value)
                #                                             .value_counts(normalize=True)[x]),
                #                                         class_instances_attribute_value))

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
            print(f"entropy parent: {measure_parent}")
            print(f"entropy attribute: {measure_attribute}")

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

    # def find_split(self):
    #     """
    #     Find the split at a given node (subset of data looked at) in a tree (global attributes and attribute values)
    #
    #     :return:
    #     """
    #
    #     tree = self.tree
    #     node = self.node
    #
    #     data_parameters = tree.data_parameters
    #     attribute_dict = tree.data_parameters.attribute_dict
    #     current_training_data = node.current_training_data_df
    #
    #     # get random attributes
    #     num_data_entries, num_attributes = current_training_data.shape
    #     random_attribute_list = data_parameters.get_random_attributes(num_attributes)
    #
    #     if INFORMATION_GAIN_DEBUG:
    #         print(f"shape of data: {current_training_data.shape}")
    #
    #     class_instances_parent = node.class_instance_partition_dict.values()
    #     num_class_instances_parent = sum(class_instances_parent)
    #
    #     # entropy_parent is the entropy of the curren node - Entropy(S)
    #     # entropy_parent = -sum(map(lambda x: (x / num_instances_parent) * log2(x / num_instances_parent)))
    #     entropy_parent = sum(map(lambda x: self.find_entropy(x, num_class_instances_parent),
    #                              class_instances_parent))
    #
    #     # TODO: store list of entropy values for each attribute
    #     entropy_attribute = {}
    #
    #     for attribute in random_attribute_list:
    #         attribute_instances = attribute_dict[attribute]
    #
    #         entropy_attribute[attribute] = entropy_parent
    #
    #         if INFORMATION_GAIN_DEBUG:
    #             print(f"Attribute: {attribute}")
    #
    #         for attribute_value in attribute_instances:
    #             # FIXME: value_counts() can use proportion instead of raw count...
    #             attribute_value_count = current_training_data[attribute].value_counts()[attribute_value]
    #             attribute_value_prop = attribute_value_count / num_data_entries
    #
    #             class_instances_attribute_value_df = \
    #                 current_training_data[current_training_data[attribute].isin([attribute_value])]
    #
    #             if INFORMATION_GAIN_DEBUG:
    #                 print(f"attribute value: {attribute_value}")
    #                 print(f"attribute proportion: {attribute_value_prop}")
    #                 print(f"current data: \n{class_instances_attribute_value_df[[CLASS_NAME, attribute]]}")
    #
    #             class_instances_attribute_value = \
    #                 get_class_instance_partition_dict(data_parameters,
    #                                                        class_instances_attribute_value_df).values()
    #             num_class_instances_attribute_value = sum(class_instances_attribute_value)
    #
    #             # FIXME: entropy values are greater than 1...
    #             # entropy_attribute[attribute] += attribute_value_prop * \
    #             #                                 self.find_entropy(class_instances_attribute_value)
    #             entropy_attribute[attribute] -= attribute_value_prop * \
    #                                             sum(map(lambda x:
    #                                                     self.find_entropy(x, num_class_instances_attribute_value),
    #                                                     class_instances_attribute_value))
    #
    #     if INFORMATION_GAIN_DEBUG:
    #         print(f"class instances: {node.class_instance_partition_dict}")
    #         print(f"entropy parent: {entropy_parent}")
    #         print(f"entropy attribute: {entropy_attribute}")
    #
    #     return max(entropy_attribute, key=entropy_attribute.get)


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

        return -(prop * log2(prop))


class MisclassificationError(InformationGain):
    def __init__(self,
                 information_gain_method: InformationGainEnum,
                 tree: Tree,
                 node: Node):
        InformationGain.__init__(self, information_gain_method, tree, node)

    def calculate_measure_partial_p(self, prop_key_value):
        pass
