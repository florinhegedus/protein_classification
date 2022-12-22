import os
import pandas as pd
import numpy as np
import cv2
import math
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, ToTensor


name_label_dict = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',   
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',   
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',   
14:  'Microtubules',
15:  'Microtubule ends',  
16:  'Cytokinetic bridge',   
17:  'Mitotic spindle',
18:  'Microtubule organizing center',  
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',   
22:  'Cell junctions', 
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',   
27:  'Rods & rings' }


class ProteinAtlasDataset(Dataset):
    def __init__(self, annotations_file, img_dir, img_size, augmentation):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.augmentation = augmentation
        self.transform = Compose([
                ToTensor(),
                RandomRotation(30),
                RandomHorizontalFlip(0.3),
                RandomVerticalFlip(0.3)
            ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = self.read_rgby(img_path, self.img_size)
        if self.augmentation:
            image = self.transform(image)
        targets = np.array([int(x) for x in self.img_labels.iloc[idx, 1].split(' ')])
        label = np.sum(np.eye(len(name_label_dict))[targets], axis=0)
        return image, label
    
    def read_rgby(self, id, size):
        colors = ['red', 'green', 'blue', 'yellow']
        flags = cv2.IMREAD_GRAYSCALE
        imgs = [cv2.resize(
                cv2.imread(id + '_' + color + '.png', flags), size).astype(
                            np.float32) / 255 for color in colors]
        imgs = np.stack(imgs, axis=-1)
        return imgs # np.moveaxis(imgs, -1, 0)

def load_data(config):
    annotations_file = os.path.join(config['data_pipe']['path'], 
                                    config['data_pipe']['annotations'])
    img_dir = os.path.join(config['data_pipe']['path'], 
                             config['data_pipe']['train_dir'])

    dataset = ProteinAtlasDataset(annotations_file=annotations_file,
                                    img_dir=img_dir,
                                    img_size=config['model']['image_size'],
                                    augmentation=config['training']['augmentation'])
    train_val_split = config['training']['train_val_split']
    no_train = math.ceil(len(dataset) * train_val_split)
    no_val = math.floor(len(dataset) * (1 - train_val_split))
    train_set, val_set = torch.utils.data.random_split(dataset, [no_train, no_val],
                                    generator=torch.Generator().manual_seed(42))
    
    train_iter = DataLoader(train_set, 
                            batch_size=config['training']['batch_size'], 
                            shuffle=True,
                            num_workers=config['training']['num_workers'])
    val_iter = DataLoader(val_set, 
                            batch_size=config['training']['batch_size'], 
                            shuffle=True,
                            num_workers=config['training']['num_workers'])
    
    return train_iter, val_iter

