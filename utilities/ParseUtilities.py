from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from parameters.Parameters import DataParameters
    from parameters.HyperParameters import HyperParameters

from utilities.DebugFlags import PARSE_UTILITIES_DEBUG

import csv
import pandas
from pandas import DataFrame
import numpy as np

# Global Attributes
ID_NAME = "id"
CLASS_NAME = "class"
MISSING_DATA_VALUE = "?"


def parse_data_training(filename):
    """
    Parse the training data csv

    :param filename: filename of the csv
    :return: data_df (DataFrame of the parsed data), output_df (DataFrame of the 'class' column), attribute_names (
    the list of attributes)
    """

    # indexes using the 'id' column (so can't access the elements of 'id' directly)
    data_df = pandas.read_csv(filename)

    data_df.pop('id')

    output_df = data_df[[CLASS_NAME]].copy()

    attribute_names = list(data_df.keys())

    return data_df, output_df, attribute_names

def parse_data_testing(filename):
    """
    Parse the testing data csv

    :param filename: filename of the csv
    :return: data_df (DataFrame of the parsed data), output_df (DataFrame of the 'class' column), attribute_names (
    the list of attributes)
    """

    # indexes using the 'id' column (so can't access the elements of 'id' directly)
    data_df = pandas.read_csv(filename)

    data_df.pop('id')

    attribute_names = list(data_df.keys())

    return data_df, attribute_names


def split_training_validation(data_df_training_total: DataFrame,
                              output_df_training_total: DataFrame,
                              hyper_parameters: HyperParameters):
    percent_training_validation = hyper_parameters.percent_training_validation

    data_df_validation = data_df_training_total.sample(frac=percent_training_validation)
    data_df_training = data_df_training_total.drop(data_df_validation.index)

    output_df_validation = output_df_training_total.loc[data_df_validation.index]
    output_df_training = output_df_validation.drop(data_df_validation.index)

    if PARSE_UTILITIES_DEBUG:
        print(f"split data shape - training: {data_df_training.shape}, validation: {data_df_validation.shape}")

    return data_df_training, output_df_training, data_df_validation, output_df_validation


# FIXME: just for testing
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


if __name__ == "__main__":
    data_df_training, output_df_training, attribute_names_training = \
        parse_data_training(f"../2023-cs429529-project1-random-forests/agaricus-lepiota - training_small.csv")

    scratch_data_df = data_df_training[:]

    # Test missing data
    print("Missing Data")
    print("before")
    print(scratch_data_df["stalk-root"])

    scratch_data_df["stalk-root"] = scratch_data_df["stalk-root"].replace(to_replace="?", value=np.NaN)
    scratch_data_df["stalk-root"] = scratch_data_df["stalk-root"].fillna(1)

    print("after")
    print(scratch_data_df["stalk-root"])

    print("original")
    print(data_df_training.loc[0:14, "stalk-root"])

    join_data_df = data_df_training[(data_df_training[CLASS_NAME] == "e") & (data_df_training["cap-shape"] == "x")]
    print(join_data_df)

    print(get_df_row_count(join_data_df, CLASS_NAME, "e"))


