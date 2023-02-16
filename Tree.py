from Parameters import DataParameters, HyperParameters
from InformationGain import InformationGainEnum
import pandas
from pandas import DataFrame

class Node:
    def __init__(self,
                 current_training_data: DataFrame,
                 parentAttribute=None,
                 parentAttributeInstance=None,
                 attribute=None,
                 class_instance_partition_dict={},
                 children_dict={},
                 output=None):
        self.current_training_data = current_training_data
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
                 hyperparameters: HyperParameters,
                 data_parameters: DataParameters):
        self.root = root
        self.hyperparameters = hyperparameters
        self.data_parameters = data_parameters
        self.frontier_list = [root]

    def grow_level(self):
        information_gain = self.hyperparameters.information_gain
        data_parameters = self.data_parameters
        frontier_list = self.frontier_list

        if information_gain == InformationGainEnum.ENTROPY:
            for node in frontier_list:
                current_training_data = node.current_training_data

                # if there are no more attributes to split with, then choose the output with the majority
                if current_training_data.empty:
                    class_instance_partition_dict = node.class_instance_partition_dict
                    node.output = max(class_instance_partition_dict, key=class_instance_partition_dict.get)

                # perform the split normally
                else:
                    print()

if __name__ == "__main__":
    print()


