from abc import abstractmethod
from enum import Enum, auto
from tree.Tree import Tree, Node


class InformationGainEnum(Enum):
    ENTROPY = auto()
    GINI_INDEX = auto()
    MISCLASSIFICATION_ERROR = auto()


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

    def find_split(self):
        tree = self.tree
        node = self.node

        data_parameters = tree.data_parameters
        attribute_dict = tree.data_parameters.attribute_dict
        current_training_data = node.current_training_data

        # get random attributes
        num_data_entries, num_attributes = current_training_data.shape
        random_attribute_list = data_parameters.get_random_attributes(num_attributes)

        # TODO: store list of entropy values for each attribute value

        for attribute in random_attribute_list:
            attribute_instances = attribute_dict[attribute]

            for attribute_value in attribute_instances:
                attribute_value_count = current_training_data[attribute].value_counts()[attribute_value]
                attribute_value_prop = attribute_value_count / num_data_entries


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
