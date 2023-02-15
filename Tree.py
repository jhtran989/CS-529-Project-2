class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children_list = []

    @classmethod
    def Leaf(cls):
        return cls(None)

class Tree:
    def __init__(self, root):
        self.root = root

