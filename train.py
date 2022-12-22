import sys
import yaml
from common.dataset import load_data
from common.utils import train, try_gpu
from models.resnet import load_model


def main():
    # Load the configuration file
    if len(sys.argv) == 1:
        raise ValueError("Please specify the config file. E.g.: \
                         python main.py config.yaml")
    config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)

    print("Loading Human Protein Atlas Dataset...")
    train_iter, val_iter = load_data(config)

    print("Loading model...")
    model = load_model(config['model'])

    print("Starting training...")
    train(model, train_iter, val_iter, num_epochs=config['training']['num_epochs'], 
            lr=config['training']['learning_rate'], 
            threshold=0.0, device=try_gpu(), 
            save_model=config['training']['save_model'])

    
if __name__ == '__main__':
    main()