from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from parameters.Parameters import DataParameters
    from tree.Tree import Node

from utilities.ParseUtilities import CLASS_NAME

from pandas import DataFrame
from toolz import valmap


def print_data_stats(node: Node, data_df: DataFrame, data_parameters: DataParameters):
    print("-------------------------------------------------")
    print("-------------------------------------------------")
    print("data:")
    print(data_df)

    if node.parentAttributeInstance is not None:
        print(data_df[[CLASS_NAME, node.parentAttribute]])

    print(data_parameters.attribute_dict)
    print(data_parameters.class_instance_list)
    print(get_class_instance_partition_dict(data_parameters, data_df))
    print(f"current attribute instance: {node.parentAttributeInstance}")

def get_df_row_count(data_df: DataFrame, column_name, column_instance):
    """
    value_counts() function of the Data Frame does NOT work if none of the rows have the attribute name

    handles exception by setting count to 0

    :param data_df:
    :param column_name:
    :param column_instance:
    :return:
    """

    try:
        count = data_df[column_name].value_counts()[column_instance]
    except:
        count = 0

    return count


def get_class_instance_partition_dict(data_parameters: DataParameters, data_df: DataFrame):
    class_instance_partition_dict = {}

    for class_instance in data_parameters.class_instance_list:
        class_instance_partition_dict[class_instance] = get_df_row_count(data_df, CLASS_NAME, class_instance)

    return class_instance_partition_dict


def get_class_instance_partition_prop_dict(class_instance_partition_dict: dict):
    total_instances = sum(class_instance_partition_dict.values())

    if total_instances == 0:
        return class_instance_partition_dict

    return valmap(lambda x: x / total_instances, class_instance_partition_dict)
