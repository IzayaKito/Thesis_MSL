import os
import pandas as pd 
import torch
from torch.utils.data import Dataset 
from skimage import io

class AerialImagesDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = rootdir
        self.transform = transform
        
    def __len__(self):
        len(self.annotations) #25000
    
    def __getitem__(self,index):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))
        
        if self.transform:
            image= self.transform(image)
        
        return (image, y_label)
    