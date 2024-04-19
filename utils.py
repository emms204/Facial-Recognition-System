import albumentations as A
import torch

class NumpyToTensor(A.ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
      image = image/255
      image = torch.from_numpy(image.transpose(2, 0, 1)).float()
      return image

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key
