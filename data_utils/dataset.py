# Python script to create datasets

import os
import numpy as np
import torch
import pandas as pd
from PIL import Image
import scipy.io
from torch.utils.data import Dataset



class StanfordCars(Dataset):
    def __init__(self, data_dir, mat_annos, data_annos, transform = None):
        self.car_annotations = pd.read_csv(data_annos)
        
        self.class_names = [name[0].replace("/", "-") for name in scipy.io.loadmat(mat_annos)["class_names"][0]]
        
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.car_annotations)
    
    def __getitem__(self, index):
        image_name = os.path.join(self.data_dir, self.car_annotations['name'][index])
        image = Image.open(image_name).convert('RGB')
        car_class = self.car_annotations['class_id'][index]
        
        if self.transform:
            image = self.transform(image)

        return image, car_class