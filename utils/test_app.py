from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import requests
import os, sys

file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(file_path)

from datasets.CIFAR10dataset import CIFAR10dataset
from torch import permute

test_ds = CIFAR10dataset(root='../data/',train=False,transform=transforms.ToTensor(),target_transform=None)
test_dl = DataLoader(test_ds,batch_size=1,shuffle=False)

#host = 'localhost'
#port = '8080'
#entrypoint = 'predict'
#url = f'http://{host}:{port}/{entrypoint}'
url = 'https://cifar10-classifier-678389556781.europe-west4.run.app/predict'

test_it = iter(test_dl)
for i in range(1,11):
    im,lbl = next(test_it)
    im_ar = np.array(im).tolist()
    response = requests.post(url,json=im_ar).json()
    print(f'Image: {i}, \n classification: {response}')