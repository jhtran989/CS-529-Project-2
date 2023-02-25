from __future__ import annotations

from enum import Enum, auto

from utilities.PrintUtilities import auto_str


@auto_str
class InformationGainEnum(Enum):
    ENTROPY = auto()
    GINI_INDEX = auto()
    MISCLASSIFICATION_ERROR = auto()

    def __str__(self):
        return self.name

def get_normalized_prob(attribute_instances_count_dict: dict, total_non_missing_data_entries):
    attribute_instances_count_values = attribute_instances_count_dict.values()

    return [attribute_value_count / total_non_missing_data_entries
            for attribute_value_count in attribute_instances_count_values]

def get_uniform_prob(attribute_instances_count_dict: dict):
    num_attribute_instances = len(attribute_instances_count_dict)

    return [1 / num_attribute_instances for _ in range(num_attribute_instances)]

