from __future__ import annotations
from typing import TYPE_CHECKING

import pandas

from utilities.TreeUtilities import get_class_instance_partition_prop_dict

if TYPE_CHECKING:
    from parameters.Parameters import DataParameters
    from parameters.HyperParameters import HyperParameters

from tree.Tree import Node, Tree

from pandas import DataFrame
import random
from math import log2

from utilities.ParseUtilities import ID_NAME, CLASS_NAME
from utilities.DebugFlags import RANDOM_FOREST_DEBUG, RANDOM_FOREST_PRINT, RANDOM_FOREST_TREE_PROGRESS, RANDOM_FOREST_FIND
from utilities.AccuracyUtilities import check_tree_data_accuracy


def calculate_measure_partial_p(prop_key_value):
    prop = prop_key_value[1]

    if prop == 0:
        return 0

    return -(prop * log2(prop))


def calculate_measure_total(class_instances_dict):
    return sum(map(
        lambda x: calculate_measure_partial_p(x),
        get_class_instance_partition_prop_dict(class_instances_dict).items()))


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

        tree_list = self.tree_list
        num_trees = hyper_parameters.num_trees

        training_num_rows, _ = training_data_df.shape
        bootstrap_num_rows = int(training_num_rows / num_trees)
        bootstrap_fraction = 1 / num_trees

        for tree_index in range(num_trees):
            current_training_data_df = training_data_df.sample(frac=bootstrap_fraction)

            if RANDOM_FOREST_TREE_PROGRESS:
                print(f"------------------------------------------")
                print(f"tree index {tree_index}...")
                print(f"bootstrap num data rows: {bootstrap_num_rows}")
                print(f"bootstrap fraction: {bootstrap_fraction}")

            root = Node(current_training_data_df)

            test_tree = Tree(root, hyper_parameters, data_parameters, validation_data_df)

            test_tree.build_tree()
            tree_list.append(test_tree)

            if RANDOM_FOREST_TREE_PROGRESS:
                print(f"tree index {tree_index} DONE")
                print(f"------------------------------------------")

    def check_random_forest_data_accuracy(self, data_df, print_stats=True, check_output=True):
        tree_list = self.tree_list

        # beginning index to create the id column
        beginning_index = 1

        num_success = 0
        total = 0
        tree_success_list = []
        tree_num_success = [0 for _ in range(len(tree_list))]
        tree_total = [0 for _ in range(len(tree_list))]
        tree_success_list_individual = []

        prediction_rows_nested_dict = {}
        id_list = []
        majority_prediction_list = []

        for tree_index, tree in enumerate(tree_list):
            if RANDOM_FOREST_TREE_PROGRESS:
                print(f"--------------------------------------------------------------")
                print(f"tree index: {tree_index}")
                print(f"--------------------------------------------------------------")

            _, tree_output_dict = check_tree_data_accuracy(data_df,
                                                           tree,
                                                           print_stats=print_stats,
                                                           check_output=check_output)

            if RANDOM_FOREST_TREE_PROGRESS:
                print(f"shape of output list: {len(tree_output_dict)}")
                print(f"shape of data: {data_df.shape}")

            for data_index, data_row in data_df.iterrows():
                # only add a dict for each row -- initially on the first tree
                if tree_index == 0:
                    prediction_rows_nested_dict[data_index] = {}

                current_prediction_dict = prediction_rows_nested_dict[data_index]
                output = tree_output_dict[data_index]

                if output in current_prediction_dict.keys():
                    current_prediction_dict[output] += 1
                else:
                    current_prediction_dict[output] = 1

        for data_index, data_row in data_df.iterrows():
            current_prediction_dict = prediction_rows_nested_dict[data_index]
            majority_output = max(current_prediction_dict, key=current_prediction_dict.get)

            entropy = calculate_measure_total(current_prediction_dict)

            if RANDOM_FOREST_FIND:
                if entropy > 0.90:
                    print(f"...data row: {data_index + beginning_index}")

            id_list.append(data_index + beginning_index)
            majority_prediction_list.append(majority_output)

            if RANDOM_FOREST_DEBUG:
                print(f"--------------------------")
                print(f"data row index: {data_index}")
                print(f"current prediction dict: {current_prediction_dict}")

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

    def check_training_data(self, print_stats=True):
        _, _ = self.check_random_forest_data_accuracy(self.training_data_df, print_stats=print_stats)

