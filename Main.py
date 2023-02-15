from Utilities \
    import parse_data_training, parse_data_testing
from Attribute \
    import Attribute, ClassificationClass, Parameters

if __name__ == "__main__":
    data_df_training, output_df_training, attribute_names_list_training = \
        parse_data_training(f"2023-cs429529-project1-random-forests/agaricus-lepiota - training.csv")

    parameters = Parameters(None, None)
    parameters.set_attribute_dict(data_df_training, attribute_names_list_training)
    parameters.set_class_instance_list(output_df_training)

    print(parameters.attribute_dict)
    print(parameters.class_instance_list)
