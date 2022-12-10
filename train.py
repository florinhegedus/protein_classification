import sys
import yaml
from common.dataset import load_data
from common.utils import train, try_gpu
from models.resnet import ResNet


def main():
    # Load the configuration file
    if len(sys.argv) == 1:
        raise ValueError("Please specify the config file. E.g.: \
                         python main.py config.yaml")
    config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)

    print("Loading Human Protein Atlas Dataset...")
    train_iter, val_iter = load_data(config)
    train_features, train_labels = next(iter(train_iter))

    print("Loading model...")
    model = ResNet(config['model'])

    print("Starting training...")
    train_loss_all, train_acc_all, val_loss_all, val_acc_all = train(model, 
                                                                    train_iter, 
                                                                    val_iter, 
                                                                    num_epochs=config['training']['num_epochs'], 
                                                                    lr=config['training']['learning_rate'], 
                                                                    device=try_gpu())
    
    
if __name__ == '__main__':
    main()