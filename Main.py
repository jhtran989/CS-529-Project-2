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
MAIN_DEBUG = False


if __name__ == "__main__":
    # training with a small subset first
    data_df_training, output_df_training, attribute_names_list_training = \
        parse_data_training(f"2023-cs429529-project1-random-forests/agaricus-lepiota - training_small.csv")

    # actual training set
    # data_df_training, output_df_training, attribute_names_list_training = \
    #     parse_data_training(f"2023-cs429529-project1-random-forests/agaricus-lepiota - training.csv")

    hyper_parameters = HyperParameters(0.95,
                                       0.05,
                                       InformationGainEnum.GINI_INDEX,
                                       [5, 7],
                                       1,
                                       0.2)

    data_parameters = DataParameters(data_df_training, output_df_training, attribute_names_list_training)

    # if MAIN_DEBUG:
    #     print("data:")
    #     print(data_df_training)
    #     print(data_parameters.attribute_dict)
    #     print(data_parameters.class_instance_list)
    #     print(get_class_instance_partition_dict(data_parameters, data_df_training))

    root = Node(data_df_training)

    test_tree = Tree(root, hyper_parameters, data_parameters)
    # test_tree.grow_level()
    test_tree.build_tree()

    # print(str(test_tree))
    # FIXME: test output of tree

    num_success = 0
    total = 0
    # iterrows automatically does enumeration
    for index, row in data_df_training.iterrows():
        # print(f"row:")
        # print(row)

        output = test_tree.get_output(row)
        print(f"row index: {index}")
        print(f"predicted: {output}")
        print(f"actual: {row[CLASS_NAME]}")

        total += 1
        if output == row[CLASS_NAME]:
            print("SUCCESS")
            num_success += 1
        else:
            print("FAIL")


    print(f"success rate: {num_success / total}")



