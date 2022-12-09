import sys
import os
import yaml
from torch.utils.data import DataLoader

from common.dataset import ProteinAtlasDataset

def main():
    # Load the configuration file
    if len(sys.argv) == 1:
        raise ValueError("Please specify the config file. E.g.: \
                         python main.py config.yaml")
    config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    
    annotations_file = os.path.join(config['data_pipe']['path'], 
                                    config['data_pipe']['annotations'])
    img_dir = os.path.join(config['data_pipe']['path'], 
                             config['data_pipe']['train_dir'])

    dataset = ProteinAtlasDataset(annotations_file=annotations_file, 
                                  img_dir=img_dir)
    for i in range(len(dataset)):
        sample, label = dataset[i]
    
    
if __name__ == '__main__':
    main()