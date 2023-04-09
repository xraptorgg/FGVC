# Python script to create datasets

import os
import numpy as np
import torch
from PIL import Image
import scipy.io
from torch.utils.data import Dataset



class StanfordCars(Dataset):
    def __init__(self, data_dir, mat_annos, data_annos, transform = None):
        self.car_annotations = scipy.io.loadmat(data_annos)["annotations"][0]
        
        self.class_names = [name[0].replace("/", "-") for name in scipy.io.loadmat(mat_annos)["class_names"][0]]
        
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.car_annotations)
    
    def __getitem__(self, index):
        image_name = os.path.join(self.data_dir, self.car_annotations[index][-1][0])
        image = Image.open(image_name).convert('RGB')
        car_class = self.car_annotations[index][-2][0][0]
        car_class = torch.from_numpy(np.array(car_class.astype(np.float32))).long() - 1
        
        if self.transform:
            image = self.transform(image)

        return image, car_class