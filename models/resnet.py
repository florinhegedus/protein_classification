import timm
import torch
from torch import nn

class ResNet(nn.Module):

    def __init__(self, model_config):

        super().__init__()

        model_type = model_config['type']
        continue_training = model_config['continue_training']

        if not continue_training:
            self.encoder = timm.create_model(model_type, num_classes=28, 
                                                in_chans=4, pretrained=True)
        else:
            self.encoder = timm.create_model(model_type, num_classes=28, 
                                                in_chans=4, pretrained=False)
        
        self.encoder = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.act1,
            self.encoder.maxpool,
            self.encoder.layer1,
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4,
            self.encoder.global_pool
        )
        self.neck = nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 28)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.neck(x)
        # x = self.sigmoid(x)
        return x

def load_model(model_config):
    model = ResNet(model_config=model_config)
    if model_config['continue_training']:
        print("Loading trained model...")
        model.load_state_dict(torch.load(model_config['weights_path']))
    return model