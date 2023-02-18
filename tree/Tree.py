from abc import abstractmethod
from math import log2

from parameters.Parameters import DataParameters
from parameters.HyperParameters import HyperParameters
from decision.InformationGain import InformationGainEnum
import pandas
from pandas import DataFrame

from Utilities import ID_NAME, CLASS_NAME


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
        self.frontier_list = [root]

        # IMPORTANT
        # class_instance_partition_dict = {}
        # for class_instance in self.data_parameters.class_instance_list:
        #     class_instance_partition_dict[class_instance] = \
        #         self.root.current_training_data.value_counts()[class_instance]

        self.root.class_instance_partition_dict = \
            self.get_class_instance_partition_dict(data_parameters, root.current_training_data_df)

    def grow_level(self):
        information_gain = self.hyper_parameters.information_gain_method
        data_parameters = self.data_parameters
        frontier_list = self.frontier_list

        # TODO: set parent stuff for each node in the frontier_list...

        if information_gain == InformationGainEnum.ENTROPY:
            for node in frontier_list:
                current_training_data = node.current_training_data_df

                # if there are no more attributes to split with, then choose the output with the majority
                if current_training_data.empty:
                    class_instance_partition_dict = node.class_instance_partition_dict
                    node.output = max(class_instance_partition_dict, key=class_instance_partition_dict.get)

                # perform the split normally
                else:
                    infromation_gain_method = self.hyper_parameters.information_gain_method

                    if infromation_gain_method == InformationGainEnum.ENTROPY:
                        information_gain = Entropy(infromation_gain_method, self, node)

                        information_gain.find_split()

    @staticmethod
    def get_class_instance_partition_dict(data_parameters: DataParameters, data_df: DataFrame):
        class_instance_partition_dict = {}

        for class_instance in data_parameters.class_instance_list:
            try:
                class_instance_partition_dict[class_instance] = \
                    data_df[CLASS_NAME].value_counts()[class_instance]
            except:
                class_instance_partition_dict[class_instance] = 0

        return class_instance_partition_dict


# FIXME: moved information stuff into tree due to CIRCULAR IMPORT...
# global variables
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
                    Tree.get_class_instance_partition_dict(data_parameters,
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
            print(f"class instances: {class_instances_parent}")
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


#################################################

if __name__ == "__main__":
    print()
