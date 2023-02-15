from Attribute import Parameters


class Node:
    def __init__(self, attribute):
        self.parentAttribute = None
        self.parentAttributeInstance = None
        self.attribute = attribute
        self.class_instance_partition_dict = {}
        self.children_dict = {}

    @classmethod
    def leaf(cls):
        return cls(None)


"""
Static objects of Node
"""
LeafNode = Node.leaf()


class Tree:
    def __init__(self, root):
        self.root = root

