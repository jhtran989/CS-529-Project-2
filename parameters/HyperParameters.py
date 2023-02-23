from utilities.InformationGainUtilities import InformationGainEnum

from utilities.PrintUtilities import auto_str


@auto_str
class HyperParameters:
    def __init__(self,
                 class_instance_cutoff_ratio,
                 chi_square_alpha,
                 information_gain_method: InformationGainEnum,
                 max_num_attributes_check,
                 num_trees,
                 max_depth_cutoff,
                 validation_cycle,
                 percent_training_validation):
        self.class_instance_cutoff_ratio = class_instance_cutoff_ratio
        self.chi_square_alpha = chi_square_alpha
        self.information_gain_method = information_gain_method
        self.max_num_attributes_check = max_num_attributes_check
        self.num_trees = num_trees
        self.max_depth_cutoff = max_depth_cutoff
        self.validation_cycle = validation_cycle
        self.percent_training_validation = percent_training_validation
