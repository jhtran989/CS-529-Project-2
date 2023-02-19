from Utilities \
    import parse_data_training
from parameters.Parameters import DataParameters
from tree.Tree import Node, Tree
from parameters.HyperParameters import HyperParameters
from decision.InformationGain import InformationGainEnum

from Utilities import CLASS_NAME
from tree.TreeUtilities import get_class_instance_partition_dict

import pandas

# Global variables
MAIN_DEBUG = Tree


if __name__ == "__main__":
    # testing with a small dataset first
    data_df_training, output_df_training, attribute_names_list_training = \
        parse_data_training(f"2023-cs429529-project1-random-forests/agaricus-lepiota - training_small.csv")

    hyper_parameters = HyperParameters(0.95,
                                       0.05,
                                       InformationGainEnum.MISCLASSIFICATION_ERROR,
                                       [5, 7],
                                       1,
                                       0.2)

    data_parameters = DataParameters(data_df_training, output_df_training, attribute_names_list_training)

    if MAIN_DEBUG:
        print("data:")
        print(data_df_training)
        print(data_parameters.attribute_dict)
        print(data_parameters.class_instance_list)
        print(get_class_instance_partition_dict(data_parameters, data_df_training))

    root = Node(data_df_training)

    test_tree = Tree(root, hyper_parameters, data_parameters)
    test_tree.grow_level()



