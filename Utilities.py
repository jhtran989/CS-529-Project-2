import csv
import pandas

# Global Attributes
ID_NAME = "id"
CLASS_NAME = "class"


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

    # FIXME: create copy of the output
    # output_df = data_df.pop('class')
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

if __name__ == "__main__":
    data_df_training, output_df_training, attribute_names_training = \
        parse_data_training(f"2023-cs429529-project1-random-forests/agaricus-lepiota - training.csv")

    print(attribute_names_training)
    print(data_df_training)
    print(data_df_training['cap-shape'][0])
    print(output_df_training)
    print(output_df_training[0])

    data_df_testing, attribute_names_testing = \
        parse_data_testing(f"2023-cs429529-project1-random-forests/agaricus-lepiota - testing.csv")

    print(data_df_testing)
    print(attribute_names_testing)
    print(data_df_testing['cap-shape'][1])

    for entry in data_df_testing['cap-shape']:
        print(f"{entry}", end=" ")
    print()

    print(data_df_testing['cap-shape'][1])

    for index, value in enumerate(data_df_training['cap-shape']):
        print(f"{index}: {value}")
