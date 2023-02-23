from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tree.Tree import Tree, Node
    from parameters.Parameters import DataParameters

from utilities.AccuracyUtilities import check_tree_data_accuracy
from utilities.DebugFlags import VALIDATION_PRINT

import math


class ValidationCheck:
    def __init__(self,
                 tree: Tree):
        self.tree = tree
        self.end_termination = False

    def check_validation(self):
        tree = self.tree

        validation_data_df = tree.validation_data_df

        if VALIDATION_PRINT:
            # print(f"root attribute: {tree.root.attribute}")
            print("validation...")

        current_validation_accuracy, _ = check_tree_data_accuracy(validation_data_df, tree, print_stats=False)

        if VALIDATION_PRINT:
            print("validation DONE")
            print(f"validation accuracy: {current_validation_accuracy}")

        # previous_validation_accuracy = tree.current_validation_accuracy
        tree.previous_validation_accuracy = max(tree.previous_validation_accuracy,
                                                tree.current_validation_accuracy)
        previous_validation_accuracy = tree.previous_validation_accuracy
        tree.current_validation_accuracy = current_validation_accuracy

        # current_validation_accuracy >= previous_validation_accuracy
        # FIXME: no absolute magnitude
        tol = 0
        validation_success = (current_validation_accuracy - previous_validation_accuracy) >= tol

        if VALIDATION_PRINT:
            if self.end_termination:
                print("validation TERMINATED")
            else:
                if validation_success:
                    print("validation SUCCESS")
                else:
                    print("validation FAILED")

        return validation_success

if __name__ == "__main__":
    b = [1, 2, 2]
    for _ in range(4):
        a = b.copy()
        a.append(2)

    print(f"{b}")

    c = set(b)

    print(f"{c}")
    print(f"{b}")
