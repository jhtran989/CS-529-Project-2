from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tree.Tree import Tree, Node

from abc import abstractmethod
from enum import Enum, auto
from math import log2

from Utilities import CLASS_NAME
from tree.TreeUtilities import get_class_instance_partition_dict


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
    def find_split(self):
        pass


class Entropy(InformationGain):
    def __init__(self,
                 information_gain_method: InformationGainEnum,
                 tree: Tree,
                 node: Node):
        InformationGain.__init__(self, information_gain_method, tree, node)

    def find_entropy(self, num_class_instances, total_instances):
        prop = num_class_instances / total_instances

        # special case where proportion is 0
        if prop == 0:
            return 0

        return -(prop * log2(prop))

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

        class_instances_parent = node.class_instance_partition_dict.values()
        num_class_instances_parent = sum(class_instances_parent)

        # entropy_parent is the entropy of the curren node - Entropy(S)
        # entropy_parent = -sum(map(lambda x: (x / num_instances_parent) * log2(x / num_instances_parent)))
        entropy_parent = sum(map(lambda x: self.find_entropy(x, num_class_instances_parent),
                                 class_instances_parent))

        # TODO: store list of entropy values for each attribute
        entropy_attribute = {}

        for attribute in random_attribute_list:
            attribute_instances = attribute_dict[attribute]

            entropy_attribute[attribute] = entropy_parent

            if INFORMATION_GAIN_DEBUG:
                print(f"Attribute: {attribute}")

            for attribute_value in attribute_instances:
                # FIXME: value_counts() can use proportion instead of raw count...
                attribute_value_count = current_training_data[attribute].value_counts()[attribute_value]
                attribute_value_prop = attribute_value_count / num_data_entries

                class_instances_attribute_value_df = \
                    current_training_data[current_training_data[attribute].isin([attribute_value])]

                if INFORMATION_GAIN_DEBUG:
                    print(f"attribute value: {attribute_value}")
                    print(f"attribute proportion: {attribute_value_prop}")
                    print(f"current data: \n{class_instances_attribute_value_df[[CLASS_NAME, attribute]]}")

                class_instances_attribute_value = \
                    get_class_instance_partition_dict(data_parameters,
                                                           class_instances_attribute_value_df).values()
                num_class_instances_attribute_value = sum(class_instances_attribute_value)

                # FIXME: entropy values are greater than 1...
                # entropy_attribute[attribute] += attribute_value_prop * \
                #                                 self.find_entropy(class_instances_attribute_value)
                entropy_attribute[attribute] -= attribute_value_prop * \
                                                sum(map(lambda x:
                                                        self.find_entropy(x, num_class_instances_attribute_value),
                                                        class_instances_attribute_value))

        if INFORMATION_GAIN_DEBUG:
            print(f"class instances: {node.class_instance_partition_dict}")
            print(f"entropy parent: {entropy_parent}")
            print(f"entropy attribute: {entropy_attribute}")

        return max(entropy_attribute, key=entropy_attribute.get)


class GiniIndex(InformationGain):
    def __init__(self,
                 information_gain_method: InformationGainEnum,
                 tree: Tree,
                 node: Node):
        InformationGain.__init__(self, information_gain_method, tree, node)


class MisclassificationError(InformationGain):
    def __init__(self,
                 information_gain_method: InformationGainEnum,
                 tree: Tree,
                 node: Node):
        InformationGain.__init__(self, information_gain_method, tree, node)
