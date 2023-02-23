from __future__ import annotations
from typing import TYPE_CHECKING

import pandas

if TYPE_CHECKING:
    from parameters.Parameters import DataParameters
    from parameters.HyperParameters import HyperParameters

from tree.Tree import Node, Tree

from pandas import DataFrame
import random

from utilities.ParseUtilities import ID_NAME, CLASS_NAME
from utilities.DebugFlags import RANDOM_FOREST_DEBUG, RANDOM_FOREST_PRINT, RANDOM_FOREST_TREE_PROGRESS
from utilities.AccuracyUtilities import check_tree_data_accuracy


class RandomForest:
    def __init__(self,
                 training_data_df: DataFrame,
                 validation_data_df: DataFrame,
                 testing_data_df: DataFrame,
                 data_parameters: DataParameters,
                 hyper_parameters: HyperParameters):

        self.training_data_df = training_data_df
        self.validation_data_df = validation_data_df
        self.testing_data_df = testing_data_df
        self.data_parameters = data_parameters
        self.hyper_parameters = hyper_parameters

        # set the tree list
        self.tree_list = []

    def get_tree_list(self):
        return self.tree_list

    def set_tree_list(self, tree_list):
        self.tree_list = tree_list


    def generate_random_forest(self):
        hyper_parameters = self.hyper_parameters
        data_parameters = self.data_parameters
        training_data_df = self.training_data_df
        validation_data_df = self.validation_data_df

        # tree_list = []
        tree_list = self.tree_list
        num_trees = hyper_parameters.num_trees

        training_num_rows, _ = training_data_df.shape
        bootstrap_num_rows = int(training_num_rows / num_trees)
        bootstrap_fraction = 1 / num_trees

        for tree_index in range(num_trees):
            # if RANDOM_FOREST_PRINT:
            #     print(f"random forest - training data shape: {training_data_df.shape}")

            # TODO: bootstrap the training data based on number of trees
            current_training_data_df = training_data_df.sample(frac=bootstrap_fraction)

            if RANDOM_FOREST_TREE_PROGRESS:
                print(f"------------------------------------------")
                print(f"tree index {tree_index}...")
                print(f"bootstrap num data rows: {bootstrap_num_rows}")
                print(f"bootstrap fraction: {bootstrap_fraction}")

            root = Node(current_training_data_df)

            test_tree = Tree(root, hyper_parameters, data_parameters, validation_data_df)

            # FIXME:
            # print(f"BEFORE attribute visited list: {test_tree.root.attribute_visited_list}")

            # test_tree.grow_level()
            test_tree.build_tree()
            tree_list.append(test_tree)

            if RANDOM_FOREST_TREE_PROGRESS:
                print(f"tree index {tree_index} DONE")
                print(f"------------------------------------------")

    def check_random_forest_data_accuracy(self, data_df, print_stats=True, check_output=True):
        tree_list = self.tree_list

        num_success = 0
        total = 0
        tree_success_list = []
        tree_num_success = [0 for _ in range(len(tree_list))]
        tree_total = [0 for _ in range(len(tree_list))]
        tree_success_list_individual = []

        # FIXME: used a nested dict -- iterrows() use the ID index (not from 0)
        prediction_rows_nested_dict = {}
        id_list = []
        majority_prediction_list = []

        for tree_index, tree in enumerate(tree_list):
            if RANDOM_FOREST_TREE_PROGRESS:
                print(f"--------------------------------------------------------------")
                print(f"tree index: {tree_index}")
                print(f"--------------------------------------------------------------")

            _, tree_output_list = check_tree_data_accuracy(data_df,
                                                           tree,
                                                           print_stats=print_stats,
                                                           check_output=check_output)

            for data_index, data_row in data_df.iterrows():
                # only add a dict for each row -- initially on the first tree
                if tree_index == 0:
                    prediction_rows_nested_dict[data_index] = {}

                # print(f"data index: {data_index}")

                current_prediction_dict = prediction_rows_nested_dict[data_index]
                output = tree_output_list[tree_index]

                if output in current_prediction_dict.keys():
                    current_prediction_dict[output] += 1
                else:
                    current_prediction_dict[output] = 1

        for data_index, data_row in data_df.iterrows():
            current_prediction_dict = prediction_rows_nested_dict[data_index]
            majority_output = max(current_prediction_dict, key=current_prediction_dict.get)

            id_list.append(data_index)
            majority_prediction_list.append(majority_output)

            if RANDOM_FOREST_DEBUG:
                print(f"--------------------------")
                print(f"data row index: {data_index}")

            if check_output:
                total += 1
                if majority_output == data_row[CLASS_NAME]:
                    if RANDOM_FOREST_DEBUG:
                        print("SUCCESS")
                        print(f"--------------------------")

                    num_success += 1
                else:
                    if RANDOM_FOREST_DEBUG:
                        print("FAIL")
                        print(f"--------------------------")

        if check_output:
            random_forest_success_rate = num_success / total

            if RANDOM_FOREST_PRINT:
                print(f"---------------------------------------------------------")
                print(f"random forest success rate: {random_forest_success_rate}")
                print(f"---------------------------------------------------------")

        return id_list, majority_prediction_list

    def check_testing_data(self, print_stats=True):
        """
        Forgot we don't have the output for the test data...

        :param print_stats:
        :return:
        """
        _, _ = self.check_random_forest_data_accuracy(self.testing_data_df, print_stats=print_stats)

    def get_random_forest_prediction(self):
        print()

    def write_output_file_testing_data(self, output_filename, print_stats=True):
        testing_data_df = self.testing_data_df

        fileoutput_data_id_list, fileoutput_data_class_list = \
            self.check_random_forest_data_accuracy(testing_data_df, print_stats=print_stats, check_output=False)

        output_data_dict = {ID_NAME: fileoutput_data_id_list, CLASS_NAME: fileoutput_data_class_list}

        fileoutput_df = pandas.DataFrame(output_data_dict)
        fileoutput_df.to_csv(output_filename, index=False)

        # fileoutput_data_id_list = []
        # fileoutput_data_class_list = []
        #
        # for data_index, data_row in testing_data_df.iterrows():
        #     fileoutput_data_id_list.append(data_index)
        #     fileoutput_data_class_list.append()

    # def random_forest_get_majority_output(self):
    #     return

    def check_training_data(self, print_stats=True):
        _, _ = self.check_random_forest_data_accuracy(self.training_data_df, print_stats=print_stats)

        # training_data_df = self.training_data_df
        # tree_list = self.tree_list
        #
        # num_success = 0
        # total = 0
        # tree_success_list = []
        # tree_num_success = [0 for _ in range(len(tree_list))]
        # tree_total = [0 for _ in range(len(tree_list))]
        # tree_success_list_individual = []
        #
        # # iterrows automatically does enumeration
        # for index, row in training_data_df.iterrows():
        #     prediction_dict = {}
        #
        #     if RANDOM_FOREST_DEBUG:
        #         print(f"--------------------------")
        #         print(f"row index: {index}")
        #         print(f"--------------------------")
        #
        #     for tree_index, tree in enumerate(tree_list):
        #         # print(f"row:")
        #         # print(row)
        #
        #         output = tree.get_output(row)
        #
        #         if RANDOM_FOREST_DEBUG:
        #             print(f"--------------------------")
        #             print(f"tree index: {tree_index}")
        #             print(f"prediction: {output}")
        #
        #         if output in prediction_dict.keys():
        #             prediction_dict[output] += 1
        #         else:
        #             prediction_dict[output] = 1
        #
        #         tree_total[tree_index] += 1
        #         if output == row[CLASS_NAME]:
        #             if RANDOM_FOREST_DEBUG:
        #                 print("SUCCESS")
        #                 print(f"--------------------------")
        #
        #             tree_num_success[tree_index] += 1
        #         else:
        #             if RANDOM_FOREST_DEBUG:
        #                 print("FAIL")
        #                 print(f"--------------------------")
        #
        #     majority_output = max(prediction_dict, key=prediction_dict.get)
        #
        #     if RANDOM_FOREST_DEBUG:
        #         print(f"--------------------------")
        #         print(f"majority output: {majority_output}")
        #         print(f"actual: {row[CLASS_NAME]}")
        #
        #     total += 1
        #     if majority_output == row[CLASS_NAME]:
        #         if RANDOM_FOREST_DEBUG:
        #             print("SUCCESS")
        #             print(f"--------------------------")
        #             print(f"--------------------------")
        #
        #         num_success += 1
        #     else:
        #         if RANDOM_FOREST_DEBUG:
        #             print("FAIL")
        #             print(f"--------------------------")
        #             print(f"--------------------------")
        #
        # if RANDOM_FOREST_PRINT:
        #     print(f"---------------------------------")
        #     print(f"---------------------------------")
        #     print(f"success rate: {num_success / total}")
        #
        #     for tree_index, tree in enumerate(tree_list):
        #         tree_success_list_individual.append(tree_num_success[tree_index] / tree_total[tree_index])
        #
        #     print(f"---------------------------------")
        #     print(f"---------------------------------")
        #     print(f"Final tree stats")
        #     for tree_index, tree in enumerate(tree_list):
        #         print(f"---------------------------------")
        #         print(f"tree index: {tree_index}")
        #         print(f"tree success: {tree_success_list_individual[tree_index]}")
        #         tree.print_stats()