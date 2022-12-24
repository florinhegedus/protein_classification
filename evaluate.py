import sys
import yaml
from common.dataset import load_data, load_test_data
from common.utils import train, test, try_gpu
from models.resnet import load_model


def main():
    # Load the configuration file
    if len(sys.argv) == 1:
        raise ValueError("Please specify the config file. E.g.: \
                         python main.py config.yaml")
    config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)

    print("Loading Human Protein Atlas Dataset...")
    test_iter, img_labels = load_test_data(config)

    print("Loading model...")
    model = load_model(model_config=config['model'], evaluate=True)

    print("Starting testing...")
    test(model, test_iter, img_labels, threshold=0.0, device=try_gpu())

    
if __name__ == '__main__':
    main()