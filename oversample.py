import sys
import yaml
import os
from common.dataset import name_label_dict
from common.oversampling import Oversampling


if __name__ == '__main__':
    # Load the configuration file
    if len(sys.argv) == 1:
        raise ValueError("Please specify the config file. E.g.: \
                         python main.py config.yaml")
    config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    annotations_file = os.path.join(config['data_pipe']['path'], 
                                    config['data_pipe']['annotations'])
    oversampling = Oversampling(annotations_file, name_label_dict)
    oversampling('oversampled.csv')