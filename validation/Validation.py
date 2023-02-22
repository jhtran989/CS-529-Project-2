from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree.Tree import Tree, Node
    from parameters.Parameters import DataParameters

class Validation:
    def __init__(self,
                 tree: Tree,
                 node: Node):
        self.tree = tree
        self.node = node

    def check_validation(self):
        tree = self.tree
        node = self.node
        validation_data_df = tree.validation_data_df


