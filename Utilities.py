import csv
import pandas

# def parse_data(filename):
#     with open(filename) as csv_file:
#         csv_reader = csv.DictReader(csv_file)

def parse_data_training(filename):
    """
    Parse the training data csv

    :param filename: filename of the csv
    :return: data_df (DataFrame of the parsed data), output_df (DataFrame of the 'class' column), attribute_names (
    the list of attributes)
    """

    # indexes using the 'id' column (so can't access the elements of 'id' directly)
    data_df = pandas.read_csv(filename,
                                index_col='id')

    output_df = data_df.pop('class')

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
    data_df = pandas.read_csv(filename,
                                index_col='id')

    attribute_names = list(data_df.keys())

    return data_df, attribute_names

if __name__ == "__main__":
    data_df_training, output_df_training, attribute_names_training = \
        parse_data_training(f"2023-cs429529-project1-random-forests/agaricus-lepiota - training.csv")

    print(attribute_names_training)
    print(data_df_training['cap-shape'][1000])
    print(output_df_training[1000])

    data_df_testing, attribute_names_testing = \
        parse_data_testing(f"2023-cs429529-project1-random-forests/agaricus-lepiota - testing.csv")

    print(attribute_names_testing)
    print(data_df_testing['cap-shape'][1])

