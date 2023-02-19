from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from parameters.Parameters import DataParameters
    from parameters.HyperParameters import HyperParameters

from decision.InformationGain import InformationGainEnum, Entropy, InformationGainFactory
from tree.TreeUtilities import get_class_instance_partition_dict

import pandas
from pandas import DataFrame

from Utilities import CLASS_NAME


class Node:
    def __init__(self,
                 current_training_data_df: DataFrame,
                 parentAttribute=None,
                 parentAttributeInstance=None,
                 attribute=None,
                 class_instance_partition_dict={},
                 children_dict={},
                 output=None):
        self.current_training_data_df = current_training_data_df
        self.parentAttribute = parentAttribute
        self.parentAttributeInstance = parentAttributeInstance
        self.attribute = attribute
        self.class_instance_partition_dict = class_instance_partition_dict
        self.children_dict = children_dict
        self.output = output

    @classmethod
    def leaf(cls):
        return cls(pandas.DataFrame())


"""
Static objects of Node
"""
LeafNode = Node.leaf()


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


    # TODO: remember to update the node in the information_gain_driver each call
    def grow_level(self):
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
                self.information_gain_driver.find_split()


# SOLVED: moved information stuff into tree due to CIRCULAR IMPORT... -> used type annotations (if TYPE_CHECKING...)

#################################################

if __name__ == "__main__":
    print()
