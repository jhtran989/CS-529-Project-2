from parameters.Parameters import DataParameters
# from tree.Tree import Node, Tree
from parameters.HyperParameters import HyperParameters
from tree.RandomForest import RandomForest

from utilities.ParseUtilities \
    import parse_data_training, parse_data_testing, split_training_validation
from utilities.InformationGainUtilities import InformationGainEnum

# Global variables
MAIN_DEBUG = False
MAIN_PRINT = True


if __name__ == "__main__":
    chi_square_alpha_list = [0.99, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]
    # chi_square_alpha_list = [0.05, 0.01]
    # chi_square_alpha_list = [0.01]

    # training the entire training set
    data_df_training_total, output_df_training_total, attribute_names_list_training = \
        parse_data_training(f"2023-cs429529-project1-random-forests/agaricus-lepiota - training.csv")

    data_df_testing, _ = \
        parse_data_testing(f"2023-cs429529-project1-random-forests/agaricus-lepiota - testing.csv")

    big_random_forest_tree_list = []

    # create a random forest (defined in the hyperparameters given below) with every possible pair of chi square
    # alphas and information gain method (given in the spec -- 7 alphas and 3 information gain methods for a total of
    # 21 random forest, a total of 210 trees)

    information_gain_list = InformationGainEnum
    # information_gain_list = [InformationGainEnum.ENTROPY, InformationGainEnum.GINI_INDEX]
    # information_gain_list = [InformationGainEnum.ENTROPY]

    for chi_square_alpha in chi_square_alpha_list:
        for information_gain_method in information_gain_list:
            if MAIN_PRINT:
                print(f"-----------------------------------------------------------------")
                print(f"chi square alpha: {chi_square_alpha}")
                print(f"information gain method: {str(information_gain_method)}")
                print(f"")

            hyper_parameters = HyperParameters(0.99,
                                               chi_square_alpha,
                                               information_gain_method,
                                               5,
                                               10,
                                               5,
                                               1000,
                                               0.5)

            data_df_training, output_df_training, data_df_validation, output_df_validation = \
                split_training_validation(data_df_training_total, output_df_training_total, hyper_parameters)

            # the data_df_training_total and output_df_training_total ONLY used to find the class instance list and attribute
            # list
            data_parameters = DataParameters(data_df_training_total,
                                             output_df_training_total,
                                             attribute_names_list_training)

            # Generate Random Forest
            random_forest = RandomForest(data_df_training,
                                         data_df_validation,
                                         data_df_testing,
                                         data_parameters,
                                         hyper_parameters)
            random_forest.generate_random_forest()
            # random_forest.check_training_data()
            random_forest.check_training_data(print_stats=False)

            # FIXME: itterrows keeps the original index (skips all over...USE DICT)

            big_random_forest_tree_list.extend(random_forest.get_tree_list())

    # find the success of ALL the trees combined (above)
    # big_random_forest = RandomForest(_, _, data_df_testing, _, _)
    # big_random_forest = RandomForest(data_df_training_total, _, data_df_testing, _, _)
    big_random_forest = RandomForest(data_df_training, _, data_df_testing, _, _)
    big_random_forest.set_tree_list(big_random_forest_tree_list)

    if MAIN_PRINT:
        print(f"-------------------------------------")
        print(f"Big Random Forest")
        print(f"-------------------------------------")

    # big_random_forest.check_training_data(print_stats=False)
    big_random_forest.write_output_file_testing_data(f"testing_output/testing_data_output.csv")





