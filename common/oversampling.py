import pandas as pd
import numpy as np
from common.dataset import name_label_dict

class Oversampling:
    def __init__(self, annotations_file, name_label_dict):
        self.img_labels = pd.read_csv(annotations_file)
        self.name_label_dict = name_label_dict
        self.samples_per_class = self.calculate_class_imbalance()
    
    def calculate_class_imbalance(self, reverse=False):
        num_samples = len(self.img_labels)
        samples_per_class = np.zeros(len(self.name_label_dict))
        for idx in range(num_samples):
            targets = np.array([int(x) for x in self.img_labels.iloc[idx, 1].split(' ')])
            label = np.sum(np.eye(len(self.name_label_dict))[targets], axis=0)
            samples_per_class += label
        samples_dict = {}
        for idx in range(len(self.name_label_dict)):
            samples_dict[idx] = int(samples_per_class[idx])
        ordered = dict(sorted(samples_dict.items(), key=lambda item:item[1], reverse=reverse))
        return ordered

    def __call__(self, file_name):
        num_duplicates = [8, 4, 2, 1]
        for i, key in enumerate(self.samples_per_class.keys()):
            print(f"--{i}-- Duplicating items of class {key}")
            for idx in range(len(self.img_labels)):
                if i < 20:
                    if self.sample_contains_key(idx, key):
                        self.duplicate_row(idx, num_duplicates[i//5])
                else:
                    break
        
        # Undersampling
        ordered = self.calculate_class_imbalance(reverse=True)
        to_drop = []
        for i, key in enumerate(ordered.keys()):
            print(f"--{i}-- Removing items of class {key}, no of items: {ordered[key]}")
            for idx in range(len(self.img_labels)):
                if ordered[key] > 10000:
                    if self.sample_contains_key(idx, key, single=True):
                        to_drop.append(idx)
                        ordered[key] -= 1
                else:
                    break
        self.img_labels.drop(to_drop, axis=0, inplace=True)
        self.img_labels.to_csv(file_name, index=False)
        print(self.calculate_class_imbalance(reverse=False))
        return self.img_labels

    def sample_contains_key(self, idx, key, single=False):
        targets = [int(x) for x in self.img_labels.iloc[idx, 1].split(' ')]
        if single and len(targets) != 1:
            return False
        if key in targets:
            return True
        return False

    def duplicate_row(self, idx, times):
        for _ in range(times):
            self.img_labels = pd.concat([self.img_labels, self.img_labels.iloc[[idx]]], 
                                                                    ignore_index=True)
            targets = [int(x) for x in self.img_labels.iloc[idx, 1].split(' ')]
        for i in targets:
            self.samples_per_class[i] += times