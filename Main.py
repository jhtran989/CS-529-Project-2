from Utilities \
    import parse_data_training
from parameters.Parameters import DataParameters
from tree.Tree import Node, Tree
from parameters.HyperParameters import HyperParameters
from decision.InformationGain import InformationGainEnum

from Utilities import CLASS_NAME

import pandas

if __name__ == "__main__":
    # testing with a small dataset first
    data_df_training, output_df_training, attribute_names_list_training = \
        parse_data_training(f"2023-cs429529-project1-random-forests/agaricus-lepiota - training_small.csv")

    hyper_parameters = HyperParameters(0.95,
                                       0.05,
                                       InformationGainEnum.ENTROPY,
                                       [5, 7],
                                       1,
                                       0.2)

    # print(output_df_training[CLASS_NAME].shape)
    # print(pandas.unique(output_df_training).shape)

    data_parameters = DataParameters(data_df_training, output_df_training, attribute_names_list_training)
    # set_attribute_dict(data_df_training, attribute_names_list_training)
    # set_class_instance_list(output_df_training)

    print(data_parameters.attribute_dict)
    print(data_parameters.class_instance_list)

    root = Node(data_df_training)

    test_tree = Tree(root, hyper_parameters, data_parameters)
    test_tree.grow_level()



