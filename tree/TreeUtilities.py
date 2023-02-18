from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from parameters.Parameters import DataParameters

from Utilities import CLASS_NAME

from pandas import DataFrame


def get_class_instance_partition_dict(data_parameters: DataParameters, data_df: DataFrame):
    class_instance_partition_dict = {}

    for class_instance in data_parameters.class_instance_list:
        try:
            class_instance_partition_dict[class_instance] = \
                data_df[CLASS_NAME].value_counts()[class_instance]
        except:
            class_instance_partition_dict[class_instance] = 0

    return class_instance_partition_dict
