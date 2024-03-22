import torch
import os
from torchvision import transforms as T
from torchvision.transforms import PILToTensor
from .dift.src.models.dift_sd import SDFeaturizer

def custom_transform(img,img_wh):
    img = img.convert('RGB')
    img = img.resize(img_wh)
    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
    return img_tensor

class DiftFeatureExtractor:
    def __init__(self,device):
        self.model = SDFeaturizer(device)
        self.img_wh = (768,768)
        self.transform = lambda img: custom_transform(img,self.img_wh)
        
    def compute_features(self,image,prompt):
        image = image.permute(2,0,1).unsqueeze(0)
        image = image.reshape(1,3,self.img_wh[0],self.img_wh[1])
        ft = self.model.forward(image,prompt=prompt)
        return ft.reshape(self.img_wh[0],self.img_wh[1],-1)