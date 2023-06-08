import os
import json
from data_prep.HippocampusDatasetLoader import LoadHippocampusData
from experiments.SupUNetExperiment import SupUNetExperiment

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = "./Task04_Hippocampus"
        self.model_config = "./config.config"
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "../out"


if __name__ == "__main__":
    # Get configuration
    c = Config()

    # Load data
    print("Loading data...")

    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)

    keys = range(len(data))
    split = dict()

    train_size = 0.7
    valid_size = 0.2
    test_size = 0.1

    split['train'] = keys[:int(train_size*len(keys))]
    split['val'] = keys[int(train_size*len(keys)):int((train_size+valid_size)*len(keys))]
    split['test'] = keys[int((train_size+valid_size)*len(keys)):]

    print(len(split['test']))

    exp = SupUNetExperiment(c, split, data)
    exp.run()

    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))
