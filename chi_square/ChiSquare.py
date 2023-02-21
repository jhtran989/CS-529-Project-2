from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree.Tree import Tree, Node
    from parameters.Parameters import DataParameters

from tree.TreeUtilities import get_df_row_count, get_class_instance_partition_prop_dict
from Utilities import CLASS_NAME

from math import pow
from scipy.stats import chi2


class ChiSquare:
    def __init__(self,
                 tree: Tree,
                 node: Node,
                 chosen_attribute):
        self.tree = tree
        self.node = node
        self.chosen_attribute = chosen_attribute

    def update_node_chosen_attribute(self, node: Node, chosen_attribute):
        self.node = node
        self.chosen_attribute = chosen_attribute

    def check_termination(self):
        """
        Some code was copied from InformationGain under the find_split() function

        :return:
        """

        chosen_attribute = self.chosen_attribute
        data_parameters = self.tree.data_parameters
        chosen_attribute_instances = data_parameters.attribute_dict[chosen_attribute]
        current_node = self.node
        current_training_data_df = current_node.current_training_data_df
        class_instance_list = data_parameters.class_instance_list
        class_instance_partition_dict = current_node.class_instance_partition_dict
        class_instance_partition_prop_dict = get_class_instance_partition_prop_dict(class_instance_partition_dict)

        # stores the count of the attribute value AND class_instance (nested dict in that order)
        attribute_values_class_count_dict = {}
        # stores the total count for each attribute value
        attribute_values_class_total_dict = {}
        for attribute_value in chosen_attribute_instances:
            # need to initialize the two dicts
            attribute_values_class_count_dict[attribute_value] = {}
            attribute_values_class_total_dict[attribute_value] = 0

            for class_instance in class_instance_list:
                data_attribute_value_class_df = \
                    current_training_data_df[
                        (current_training_data_df[chosen_attribute] == attribute_value)
                        & (current_training_data_df[CLASS_NAME] == class_instance)]

                attribute_value_class_count = get_df_row_count(data_attribute_value_class_df,
                                                               chosen_attribute,
                                                               attribute_value)
                attribute_values_class_count_dict[attribute_value][class_instance] = attribute_value_class_count
                attribute_values_class_total_dict[attribute_value] += attribute_value_class_count

        # total_num_attribute_instances = sum(attribute_instances_count_dict.values())

        critical_value = 0
        for attribute_value in chosen_attribute_instances:
            for class_instance in class_instance_list:
                actual = attribute_values_class_count_dict[attribute_value][class_instance]
                expected = class_instance_partition_prop_dict[class_instance] * \
                           attribute_values_class_total_dict[attribute_value]

                # FIXME: case where expected is 0...
                # ignore the case for now
                if expected == 0:
                    pass
                else:
                    critical_value += \
                        pow(actual - expected, 2) / expected

        num_attribute_values = len(chosen_attribute_instances)
        num_class_instances = len(class_instance_list)
        degrees_of_freedom = (num_attribute_values - 1) * (num_class_instances - 1)

        hyper_parameters = self.tree.hyper_parameters
        chi_square_alpha = hyper_parameters.chi_square_alpha

        return critical_value > chi2.ppf(1 - chi_square_alpha, degrees_of_freedom)
