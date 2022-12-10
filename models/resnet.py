import timm
from torch import nn

class ResNet(nn.Module):

    def __init__(self, model_config):

        super().__init__()

        model_type = model_config['type']
        continue_training = model_config['continue_training']
        weights_path = model_config['weights_path']

        if not continue_training:
            self.encoder = timm.create_model(model_type, num_classes=28, 
                                                in_chans=4, pretrained=True)
        else:
            self.encoder = timm.create_model(model_type, num_classes=28, 
                                                in_chans=4, pretrained=False,
                                                checkpoint_path=weights_path)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.sigmoid(x)
        return x