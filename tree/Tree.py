from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parameters.Parameters import DataParameters
    from parameters.HyperParameters import HyperParameters

from decision.InformationGain import InformationGainFactory
from utilities.TreeUtilities import print_data_stats, get_class_instance_partition_dict
from chi_square.ChiSquare import ChiSquare
from validation.ValidationCheck import ValidationCheck

import pandas
from pandas import DataFrame
import random

from utilities.ParseUtilities import MISSING_DATA_VALUE
from utilities.PrintUtilities import auto_str

from utilities.DebugFlags import TREE_DEBUG, TREE_PRINT


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
                 tree_level=0,
                 output=None,
                 cached_output=None):
        self.current_training_data_df = current_training_data_df
        self.parentNode = parentNode
        self.parentAttribute = parentAttribute
        self.parentAttributeInstance = parentAttributeInstance
        self.attribute = attribute
        self.class_instance_partition_dict = class_instance_partition_dict
        self.children_dict = children_dict
        self.attribute_visited_list = attribute_visited_list
        self.tree_level = tree_level
        self.output = output
        self.cached_output = cached_output

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
                 data_parameters: DataParameters,
                 validation_data_df: DataFrame):
        self.root = root
        self.hyper_parameters = hyper_parameters
        self.data_parameters = data_parameters
        self.validation_data_df = validation_data_df

        # TODO: set other parameters
        self.frontier_list = [root]
        self.root.class_instance_partition_dict = \
            get_class_instance_partition_dict(data_parameters, root.current_training_data_df)
        self.information_gain_driver = InformationGainFactory(hyper_parameters.information_gain_method, self, None)

        # TODO: set some stats (maybe with hyper parameters)
        # level starts from 0 at the root
        self.sum_levels = 0
        self.max_depth = 0
        self.average_depth = 0
        self.total_num_nodes = 1

        # remember to update the current node AND chosen attribute at each split below
        self.chi_square = ChiSquare(self, None, None)

        # keep track of the previous and current validation accuracy for tuning hyper parameters
        self.validation_check = ValidationCheck(self)
        self.current_validation_accuracy = 0
        self.previous_validation_accuracy = 0

    def print_stats(self):
        print(f"top level attribute: {self.root.attribute}")
        print(f"sum of levels: {self.sum_levels}")
        print(f"max depth: {self.max_depth}")
        print(f"average_depth: {self.average_depth}")
        print(f"total number of nodes: {self.total_num_nodes}")

    def get_output(self, data_row_df):
        current_node = self.root
        current_attribute = current_node.attribute
        current_output = current_node.output

        while current_output is None:
            # get the next node by following the tree
            # FIXME: if a split with a MISSING DATA VALUE in the row is encountered, get a random value from the
            #  list of all possible attribute values
            current_attribute_value = data_row_df[current_attribute][0]
            if current_attribute_value == MISSING_DATA_VALUE:
                current_attribute_dict = self.data_parameters.attribute_dict
                current_attribute_value_list = current_attribute_dict[current_attribute]

                # important - sample returns a list so need to index into list
                current_attribute_value = random.sample(current_attribute_value_list, k=1)[0]

            current_node = current_node.children_dict[current_attribute_value]

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

            new_tree_level = node.tree_level + 1

            self.sum_levels += new_tree_level
            self.total_num_nodes += 1
            self.max_depth = max(self.max_depth, new_tree_level)

            # FIXME: set current output to the same as the parent initially for VALIDATION (not all frontier nodes
            #  will have their final outputs set when validation occurs)
            temporary_output = node.output

            # print(f"TEMPORARY {temporary_output}")
            # print(f"A {temporary_output}")
            # print(f"INSIDE node output {node.output}")

            assert temporary_output is not None

            children_dict[attribute_value_to_node] = Node(new_current_training_data, parentNode=node,
                                                          parentAttribute=attribute,
                                                          parentAttributeInstance=attribute_value_to_node,
                                                          class_instance_partition_dict=new_class_instance_partition_dict,
                                                          attribute_visited_list=new_attribute_visited_list,
                                                          tree_level=new_tree_level,
                                                          output=temporary_output,
                                                          cached_output=temporary_output)

        # FIXME: change the parent node output to None AFTER all the children are created (for VALIDATION)
        # SOLVED: outside the for loop
        # SOLVED: cached the output for use with EARLY VALIDATION TERMINATION
        node.output = None

        return children_dict

    def build_tree(self):
        information_gain = self.hyper_parameters.information_gain_method
        data_parameters = self.data_parameters
        frontier_list = self.frontier_list
        hyper_parameters = self.hyper_parameters
        validation_check = self.validation_check

        # TODO: set parent stuff for each node in the frontier_list...
        # while the frontier is not empty
        while frontier_list:
            update_children_dict = False
            add_new_frontier_nodes = False
            chosen_attribute = None

            node = frontier_list.pop()
            current_training_data = node.current_training_data_df
            self.information_gain_driver.update_node(node)

            if TREE_PRINT:
                print_data_stats(node, current_training_data, data_parameters)

            class_instance_partition_dict = node.class_instance_partition_dict
            class_instance_max = max(class_instance_partition_dict, key=class_instance_partition_dict.get)

            # end the termination and set the output same as the PARENT (since that produced the better results
            # during validation)
            validation_end_termination = validation_check.end_termination
            if validation_end_termination:
                node.attribute = "Early Validation Termination"

                # cached output will NEVER be None with how the tree is set up
                assert node.parentNode.cached_output is not None
                node.output = node.parentNode.cached_output

            # if there is no more data to split with, then choose the output with the majority
            elif current_training_data.empty:
                if TREE_DEBUG:
                    print("no more data...")
                    print(f"class partition dict: {class_instance_partition_dict}")
                    print(f"set output: {class_instance_max}")

                node.attribute = "No More Data"
                node.output = class_instance_max
                node.cached_output = class_instance_max

            # perform the split normally
            else:
                current_training_data_rows, current_training_data_columns = current_training_data.shape

                class_max_prop = class_instance_partition_dict[class_instance_max] / current_training_data_rows

                # TODO: need to check validation - just decrease max depth by 1 for testing
                max_depth_cutoff = self.hyper_parameters.max_depth_cutoff
                current_tree_level = node.tree_level

                if TREE_DEBUG:
                    print("getting split...")
                    print(f"current data shape: {current_training_data.shape}")
                    print(f"max class instance: {class_instance_max}")
                    print(f"max class instance frequency: {class_instance_partition_dict[class_instance_max]}")
                    print(f"total data instances: {current_training_data_rows}")

                # check for the cut-off first (hyperparameter)
                if class_max_prop >= self.hyper_parameters.class_instance_cutoff_ratio:
                    if TREE_DEBUG:
                        print("cutoff reached...")

                    node.attribute = "Class Majority Cutoff"
                    node.output = class_instance_max
                    node.cached_output = class_instance_max

                # check for the max depth (hyperparameter)
                elif current_tree_level >= max_depth_cutoff:
                    node.attribute = "Max Depth Cutoff"
                    node.output = class_instance_max
                    node.cached_output = class_instance_max

                # find the attribute with the highest information gain for the split and add nodes for each of the
                # attribute values
                else:
                    chosen_attribute = self.information_gain_driver.find_split()

                    self.chi_square.update_node_chosen_attribute(node, chosen_attribute)

                    if TREE_DEBUG:
                        print("using information gain...")
                        print(f"chosen attribute: {chosen_attribute}")

                    # TODO: implement Chi Square to check for termination (set output)
                    if self.chi_square.check_termination():
                        # # TODO: update the frontier nodes with new data and updating node parameters
                        # # FIXME: move to AFTER validation for the children dict (output is still None...)
                        node.attribute = chosen_attribute
                        node.attribute_visited_list.append(chosen_attribute)

                        # FIXME: during validation, one of the children with an INCOMPLETE path may not have an
                        #  output...
                        # FIXME: temporarily set output to the max class instance
                        node.output = class_instance_max
                        node.cached_output = class_instance_max

                        # # FIXME:
                        # print(f"node output {node.output}")

                        node.children_dict = self.get_children_dict(node, chosen_attribute)
                        update_children_dict = True

                        # FIXME: check if there is at least one attribute left to check
                        num_attributes_visited = len(data_parameters.attribute_list)
                        total_num_attributes = len(node.attribute_visited_list)

                        # no more attributes to check...
                        # FIXME: should never be True as long as max_depth (hyperparameters) is less than the
                        #  total number of attributes
                        if num_attributes_visited == total_num_attributes:
                            node.attribute = "No More Attributes to Check Along Path"
                            node.output = class_instance_max
                            node.cached_output = class_instance_max
                        else:
                            # TODO: add the children to the frontier
                            # FIXME: need to delay adding the new nodes to the frontier list until AFTER validation
                            frontier_list.extend(node.children_dict.values())
                            add_new_frontier_nodes = True

                    # the chi-square failed at the test statistic was less than the critical value
                    else:
                        node.attribute = "Chi Square Cutoff"
                        node.output = class_instance_max
                        node.cached_output = class_instance_max

            # perform validation every so often defined by validation_cycle (hyperparameter)
            # performed cyclically after a certain number of nodes are created
            # FIXME: only check after the FIRSY cycle (validation_cycle number of nodes are already in the tree)
            # FIXME: move validation at the END of the node creation (so no lingering nodes still exist in
            #  the tree WITHOUT AN OUTPUT)

            # if not validation_end_termination:

            validation_cycle = hyper_parameters.validation_cycle
            total_num_nodes = self.total_num_nodes
            if (total_num_nodes % validation_cycle) == 0 and total_num_nodes != 0:
                if not validation_check.check_validation():
                    # FIXME: reduce max depth as a simple condition for now (remember to take max so the depth
                    #  is at least 1)
                    # SOLVED: actually, just do an early termination of the tree if this happens...
                    # max_depth_cutoff = hyper_parameters.max_depth_cutoff
                    # hyper_parameters.max_depth_cutoff = max(1, max_depth_cutoff - 1)
                    hyper_parameters.max_depth_cutoff = 1
                    validation_check.end_termination = True

                    # node.attribute = "Validation Fail"
                    # node.output = class_instance_max
                    # break

                # remove current node from frontier
                # frontier_list.pop(0)

            # # TODO: update the frontier nodes with new data and updating node parameters
            # # FIXME: move to AFTER validation
            # if update_children_dict:
            #     node.children_dict = self.get_children_dict(node, chosen_attribute)
            #
            #     # TODO: add the children to the frontier
            #     # FIXME: need to delay adding the new nodes to the frontier list until AFTER validation
            #     if add_new_frontier_nodes:
            #         frontier_list.extend(node.children_dict.values())

        # at the very end of the build process
        self.average_depth = self.sum_levels / self.total_num_nodes

# SOLVED: moved information stuff into tree due to CIRCULAR IMPORT... -> used type annotations (if TYPE_CHECKING...)

#################################################

if __name__ == "__main__":
    print()
