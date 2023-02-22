# from __future__ import annotations
# from typing import TYPE_CHECKING
#
# if TYPE_CHECKING:
#     from parameters.Parameters import DataParameters
#     from parameters.HyperParameters import HyperParameters
from __future__ import annotations

from enum import Enum, auto

from utilities.PrintUtilities import auto_str


# from pandas import DataFrame
#
# from Utilities import MISSING_DATA_VALUE


# def get_normalized_probabilities(data_df: DataFrame, attribute, attribute_instances_array):
#     data_column = data_df[attribute]
#
#     try:
#         count = data_df[attribute].drop(data_column[data_df[attribute].isin(MISSING_DATA_VALUE)]).value_counts()
#     except:
#         count = 0

@auto_str
class InformationGainEnum(Enum):
    ENTROPY = auto()
    GINI_INDEX = auto()
    MISCLASSIFICATION_ERROR = auto()

def get_normalized_prob(attribute_instances_count_dict: dict, total_non_missing_data_entries):
    attribute_instances_count_values = attribute_instances_count_dict.values()

    return [attribute_value_count / total_non_missing_data_entries
            for attribute_value_count in attribute_instances_count_values]

def get_uniform_prob(attribute_instances_count_dict: dict):
    num_attribute_instances = len(attribute_instances_count_dict)

    return [1 / num_attribute_instances for _ in range(num_attribute_instances)]

