from enum import Enum, auto

# TODO: get attributes from file or hardcode them...
class AttributeEnum(Enum):
    NotImplementedError

class Attribute:
    def __init__(self, attribute, labels):
        self.attribute = attribute
        self.labels = labels


class ClassificationClass:
    def __init__(self, raw_class, instance_class):
        # TODO
        NotImplementedError

