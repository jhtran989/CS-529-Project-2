from __future__ import annotations
from typing import TYPE_CHECKING

from DebugFlags import TREE_DEBUG, TREE_PRINT

if TYPE_CHECKING:
    from parameters.Parameters import DataParameters
    from parameters.HyperParameters import HyperParameters

from decision.InformationGain import InformationGainFactory
from tree.TreeUtilities import print_data_stats, get_class_instance_partition_dict
from chi_square.ChiSquare import ChiSquare

import pandas
from pandas import DataFrame

from PrintUtilities import auto_str


@auto_str
class Node:
    def __init__(self,
                 current_training_data_df: DataFrame,
                 parentNode: Node = None,
                 parentAttribute=None,
                 parentAttributeInstance=None,
                 attribute=None,
                 class_instance_partition_dict: dict = {},
                 children_dict: dict = {},
                 attribute_visited_list: list = [],
                 output=None):
        self.current_training_data_df = current_training_data_df
        self.parentNode = parentNode
        self.parentAttribute = parentAttribute
        self.parentAttributeInstance = parentAttributeInstance
        self.attribute = attribute
        self.class_instance_partition_dict = class_instance_partition_dict
        self.children_dict = children_dict
        self.attribute_visited_list = attribute_visited_list
        self.output = output

    @classmethod
    def leaf(cls):
        return cls(pandas.DataFrame())


"""
Static objects of Node
"""
LeafNode = Node.leaf()


@auto_str
class Tree:
    def __init__(self,
                 root: Node,
                 hyper_parameters: HyperParameters,
                 data_parameters: DataParameters):
        self.root = root
        self.hyper_parameters = hyper_parameters
        self.data_parameters = data_parameters

        # TODO: set other parameters
        self.frontier_list = [root]
        self.root.class_instance_partition_dict = \
            get_class_instance_partition_dict(data_parameters, root.current_training_data_df)
        self.information_gain_driver = InformationGainFactory(hyper_parameters.information_gain_method, self, None)

        # remember to update the current node AND chosen attribute at each split below
        self.chi_square = ChiSquare(self, None, None)

    def get_output(self, data_row_df):
        current_node = self.root
        current_attribute = current_node.attribute
        current_output = current_node.output

        while current_output is None:
            # get the next node by following the tree
            current_node = current_node.children_dict[data_row_df[current_attribute][0]]

            # update the attributes and corresponding output
            current_attribute = current_node.attribute
            current_output = current_node.output

        return current_output


    # TODO: remember to update the node in the information_gain_driver each call
    def grow_level(self):
        """
        FIXME: DEPRECATED...

        :return:
        """
        information_gain = self.hyper_parameters.information_gain_method
        data_parameters = self.data_parameters
        frontier_list = self.frontier_list

        # TODO: set parent stuff for each node in the frontier_list...
        for node in frontier_list:
            current_training_data = node.current_training_data_df
            self.information_gain_driver.update_node(node)

            # if there are no more attributes to split with, then choose the output with the majority
            if current_training_data.empty:
                class_instance_partition_dict = node.class_instance_partition_dict
                node.output = max(class_instance_partition_dict, key=class_instance_partition_dict.get)

            # perform the split normally
            else:
                chosen_attribute = self.information_gain_driver.find_split()

                # TODO: update the frontier nodes with new data and updating node parameters
                node.attribute = chosen_attribute
                node.children_dict = self.get_children_dict(node, chosen_attribute)

                frontier_list.pop()

    def get_children_dict(self, node: Node, attribute):
        children_dict = {}
        attribute_values = self.data_parameters.attribute_dict[attribute]
        current_training_data = node.current_training_data_df

        for attribute_value_to_node in attribute_values:
            # SOLVED: need to check for a clear split (also use cut-off) -> remove data with attribute value

            # SOLVED: error dropping rows -> need .index
            new_current_training_data = \
                current_training_data.drop(current_training_data[
                                               current_training_data[
                                                   attribute] == attribute_value_to_node].index, axis=0)

            # if TREE_DEBUG:
            #     print("data to drop:")
            #     print(new_current_training_data)
            #     # print("previous data:")
            #     # print(current_training_data)

            # new_current_training_data = \
            #     current_training_data[
            #         current_training_data[
            #             attribute] == attribute_value_to_node]

            # get parameters for the new nodes
            new_class_instance_partition_dict = get_class_instance_partition_dict(
                self.data_parameters,
                new_current_training_data)

            # FIXME: need to make a copy of the list
            new_attribute_visited_list = node.attribute_visited_list.copy()
            new_attribute_visited_list.append(attribute)

            children_dict[attribute_value_to_node] = Node(new_current_training_data, parentNode=node,
                                                          parentAttribute=attribute,
                                                          parentAttributeInstance=attribute_value_to_node,
                                                          class_instance_partition_dict=
                                                          new_class_instance_partition_dict,
                                                          attribute_visited_list=new_attribute_visited_list)

        return children_dict

    def build_tree(self):
        information_gain = self.hyper_parameters.information_gain_method
        data_parameters = self.data_parameters
        frontier_list = self.frontier_list

        # TODO: set parent stuff for each node in the frontier_list...
        # while the frontier is not empty
        while frontier_list:
            node = frontier_list.pop()
            current_training_data = node.current_training_data_df
            self.information_gain_driver.update_node(node)

            if TREE_PRINT:
                print_data_stats(node, current_training_data, data_parameters)

            class_instance_partition_dict = node.class_instance_partition_dict
            class_instance_max = max(class_instance_partition_dict, key=class_instance_partition_dict.get)

            # if there is no more data to split with, then choose the output with the majority
            if current_training_data.empty:
                if TREE_DEBUG:
                    print("no more data...")
                    print(f"class partition dict: {class_instance_partition_dict}")
                    print(f"set output: {class_instance_max}")

                node.output = class_instance_max

            # perform the split normally
            else:
                # check for the cut-off first
                current_training_data_rows, current_training_data_columns = current_training_data.shape

                if TREE_DEBUG:
                    print("getting split...")
                    print(f"current data shape: {current_training_data.shape}")
                    print(f"max class instance: {class_instance_max}")
                    print(f"max class instance frequency: {class_instance_partition_dict[class_instance_max]}")
                    print(f"total data instances: {current_training_data_rows}")

                class_max_prop = class_instance_partition_dict[class_instance_max] / current_training_data_rows

                if class_max_prop >= self.hyper_parameters.class_instance_cutoff_ratio:
                    if TREE_DEBUG:
                        print("cutoff reached...")

                    node.output = class_instance_max

                # find the attribute with the highest information gain for the split and add nodes for each of the
                # attribute values
                else:
                    if TREE_DEBUG:
                        print("using information gain...")

                    chosen_attribute = self.information_gain_driver.find_split()

                    if TREE_DEBUG:
                        print(f"chosen attribute: {chosen_attribute}")

                    self.chi_square.update_node_chosen_attribute(node, chosen_attribute)

                    # TODO: implement Chi Square to check for termination (set output)
                    if self.chi_square.check_termination():
                        # TODO: update the frontier nodes with new data and updating node parameters
                        node.attribute = chosen_attribute
                        node.attribute_visited_list.append(chosen_attribute)
                        node.children_dict = self.get_children_dict(node, chosen_attribute)

                        # FIXME: check if there is at least one attribute left to check
                        num_attributes_visited = len(data_parameters.attribute_list)
                        total_num_attributes = len(node.attribute_visited_list)

                        # no more attributes to check...
                        if num_attributes_visited == total_num_attributes:
                            node.output = class_instance_max
                        else:
                            # TODO: add the children to the frontier
                            frontier_list.extend(node.children_dict.values())

                    # the chi-square failed at the test statistic was less than the critical value
                    else:
                        node.output = class_instance_max

                # remove current node from frontier
                # frontier_list.pop(0)


# SOLVED: moved information stuff into tree due to CIRCULAR IMPORT... -> used type annotations (if TYPE_CHECKING...)

#################################################

if __name__ == "__main__":
    print()
