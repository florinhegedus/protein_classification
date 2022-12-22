import pandas as pd
import numpy as np
from common.dataset import name_label_dict

class Oversampling:
    def __init__(self, annotations_file, name_label_dict):
        self.img_labels = pd.read_csv(annotations_file)
        self.name_label_dict = name_label_dict
        self.samples_per_class = self.calculate_class_imbalance()
    
    def calculate_class_imbalance(self):
        num_samples = len(self.img_labels)
        samples_per_class = np.zeros(len(self.name_label_dict))
        for idx in range(num_samples):
            targets = np.array([int(x) for x in self.img_labels.iloc[idx, 1].split(' ')])
            label = np.sum(np.eye(len(self.name_label_dict))[targets], axis=0)
            samples_per_class += label
        samples_dict = {}
        for idx in range(len(self.name_label_dict)):
            samples_dict[idx] = int(samples_per_class[idx])
        ordered = dict(sorted(samples_dict.items(), key=lambda item:item[1], reverse=False))
        return ordered

    def __call__(self, file_name):
        num_duplicates = [8, 4, 1]
        n, i = 0, 0
        for key in self.samples_per_class.keys():
            print(n)
            n += 1
            if n % 7 == 0:
                i += 1
            for idx in range(len(self.img_labels)):
                if n < 20:
                    if self.sample_contains_key(idx, key):
                        self.duplicate_row(idx, num_duplicates[i])
                else:
                    break
        self.img_labels.to_csv(file_name, index=False)
        return self.img_labels

    def sample_contains_key(self, idx, key):
        targets = [int(x) for x in self.img_labels.iloc[idx, 1].split(' ')]
        most_freq_classes = [0, 25, 21]
        intersection = [x for x in targets if x in most_freq_classes]
        if key in targets and len(intersection) == 0:
            return True
        return False

    def duplicate_row(self, idx, times):
        for _ in range(times):
            self.img_labels = pd.concat([self.img_labels, self.img_labels.iloc[[idx]]], 
                                                                    ignore_index=True)
            targets = [int(x) for x in self.img_labels.iloc[idx, 1].split(' ')]
        for i in targets:
            self.samples_per_class[i] += times