from decision.InformationGain import InformationGainEnum


class HyperParameters:
    def __init__(self,
                 class_instance_cutoff_ratio,
                 chi_square_alpha,
                 information_gain_method: InformationGainEnum,
                 num_attributes_interval,
                 num_trees,
                 percent_training_validation):
        self.class_instance_cutoff_ratio = class_instance_cutoff_ratio
        self.chi_square_alpha = chi_square_alpha
        self.information_gain_method = information_gain_method
        self.num_attributes_interval = num_attributes_interval
        self.num_trees = num_trees
        self.percent_training_validation = percent_training_validation
