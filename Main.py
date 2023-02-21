from Utilities \
    import parse_data_training
from parameters.Parameters import DataParameters
# from tree.Tree import Node, Tree
from parameters.HyperParameters import HyperParameters
from decision.InformationGain import InformationGainEnum
from tree.RandomForest import RandomForest

from Utilities import CLASS_NAME
from tree.TreeUtilities import get_class_instance_partition_dict

import pandas

# Global variables
MAIN_DEBUG = False


if __name__ == "__main__":
    # training with a small subset first
    data_df_training, output_df_training, attribute_names_list_training = \
        parse_data_training(f"2023-cs429529-project1-random-forests/agaricus-lepiota - training_small.csv")

    # training with a medium subset next
    # data_df_training, output_df_training, attribute_names_list_training = \
    #     parse_data_training(f"2023-cs429529-project1-random-forests/agaricus-lepiota - training_medium.csv")

    # actual training set
    # data_df_training, output_df_training, attribute_names_list_training = \
    #     parse_data_training(f"2023-cs429529-project1-random-forests/agaricus-lepiota - training.csv")

    hyper_parameters = HyperParameters(0.95,
                                       0.05,
                                       InformationGainEnum.GINI_INDEX,
                                       5,
                                       3,
                                       0.2)
    # hyper_parameters = HyperParameters(0.90,
    #                                    0.01,
    #                                    InformationGainEnum.GINI_INDEX,
    #                                    5,
    #                                    1,
    #                                    0.2)

    data_parameters = DataParameters(data_df_training, output_df_training, attribute_names_list_training)

    # if MAIN_DEBUG:
    #     print("data:")
    #     print(data_df_training)
    #     print(data_parameters.attribute_dict)
    #     print(data_parameters.class_instance_list)
    #     print(get_class_instance_partition_dict(data_parameters, data_df_training))

    # Generate Random Forest
    random_forest = RandomForest(data_df_training, data_parameters, hyper_parameters)
    random_forest.generate_random_forest()
    random_forest.check_training_data()

    # tree_list = []
    # for _ in range(hyper_parameters.num_trees):
    #     root = Node(data_df_training)
    #
    #     test_tree = Tree(root, hyper_parameters, data_parameters)
    #     # test_tree.grow_level()
    #     test_tree.build_tree()
    #     tree_list.append(test_tree)
    #
    # # print(str(test_tree))
    # # FIXME: test output of tree
    #
    # num_success = 0
    # total = 0
    # # iterrows automatically does enumeration
    # for index, row in data_df_training.iterrows():
    #     prediction_dict = {}
    #
    #     print(f"row index: {index}")
    #
    #     for tree_index, tree in enumerate(tree_list):
    #         # print(f"row:")
    #         # print(row)
    #
    #         output = tree.get_output(row)
    #
    #         print(f"tree index: {tree_index}")
    #         print(f"prediction: {output}")
    #
    #         if output in prediction_dict.keys():
    #             prediction_dict[output] += 1
    #         else:
    #             prediction_dict[output] = 1
    #
    #     majority_output = max(prediction_dict, key=prediction_dict.get)
    #
    #     print(f"majority output: {majority_output}")
    #     print(f"actual: {row[CLASS_NAME]}")
    #
    #     total += 1
    #     if majority_output == row[CLASS_NAME]:
    #         print("SUCCESS")
    #         num_success += 1
    #     else:
    #         print("FAIL")
    #
    # print(f"---------------------------------")
    # print(f"---------------------------------")
    # print(f"success rate: {num_success / total}")
    #
    # print(f"---------------------------------")
    # print(f"---------------------------------")
    # print(f"Final tree stats")
    # for tree_index, tree in enumerate(tree_list):
    #     print(f"---------------------------------")
    #     print(f"tree index: {tree_index}")
    #     tree.print_stats()



