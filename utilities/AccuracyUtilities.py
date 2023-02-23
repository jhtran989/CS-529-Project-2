from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from parameters.Parameters import DataParameters
    from parameters.HyperParameters import HyperParameters
    from tree.Tree import Tree

from utilities.DebugFlags import ACCURACY_UTILITIES_DEBUG, ACCURACY_UTILITIES_PRINT, VALIDATION_DEBUG, VALIDATION_SWITCH
from utilities.ParseUtilities import CLASS_NAME

from pandas import DataFrame


def check_tree_data_accuracy(data_df: DataFrame, current_tree: Tree, print_stats=True, check_output=True):
    num_success = 0
    total = 0
    tree = current_tree
    output_dict = {}
    tree_success_rate = 0

    # iterrows automatically does enumeration
    for data_index, data_row in data_df.iterrows():
        prediction_dict = {}

        if ACCURACY_UTILITIES_DEBUG:
            print(f"--------------------------")
            print(f"row index: {data_index}")
            print(f"--------------------------")

        output = tree.get_output(data_row)

        size_before = len(output_dict)
        output_dict[data_index] = output
        size_after = len(output_dict)

        if ACCURACY_UTILITIES_DEBUG:
            print(f"--------------------------")
            print(f"prediction: {output}")

        if check_output:
            total += 1
            if output == data_row[CLASS_NAME]:
                if ACCURACY_UTILITIES_DEBUG:
                    print("SUCCESS")
                    print(f"--------------------------")

                num_success += 1
            else:
                if ACCURACY_UTILITIES_DEBUG:
                    print("FAIL")
                    print(f"--------------------------")

    # print(f"output list size: {len(output_dict)}")
    # print(f"size of data: {data_df.shape}")

    if check_output:
        tree_success_rate = num_success / total

        if print_stats:
            print(f"--------------------------------------------------------------")
            print(f"success rate: {tree_success_rate}")
            print(f"---------------------------------")
            print(f"Final tree stats")
            print(f"--------------------------------------------------------------")
            tree.print_stats()


    return tree_success_rate, output_dict