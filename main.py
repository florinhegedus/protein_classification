import sys
import yaml
from common.dataset import load_data


def main():
    # Load the configuration file
    if len(sys.argv) == 1:
        raise ValueError("Please specify the config file. E.g.: \
                         python main.py config.yaml")
    config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)

    train_iter, val_iter = load_data(config)
    
    train_features, train_labels = next(iter(train_iter))
    print(train_features.shape)
    print(train_labels.shape)
    
    
if __name__ == '__main__':
    main()