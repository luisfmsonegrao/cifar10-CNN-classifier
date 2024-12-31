from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import requests
import os, sys
import json

file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(file_path)

from datasets.CIFAR10dataset import CIFAR10dataset
from torch import permute

test_ds = CIFAR10dataset(root='../data/',train=False,transform=transforms.ToTensor(),target_transform=None)
test_dl = DataLoader(test_ds,batch_size=1,shuffle=False)

#host = 'localhost' # use with local container running torchserve
#port = '9696'
entrypoint = 'predictions/cifar10-classifier' #use with torchserve
#url = f'http://{host}:{port}/{entrypoint}'
url = f'https://cifar10-classifier-678389556781.europe-west4.run.app/{entrypoint}' #run with GCP container running torchserve

test_it = iter(test_dl)
for i in range(1,11):
    im,lbl = next(test_it)
    im_ar = np.array(im).tolist()
    response = requests.post(url,data=json.dumps(im_ar)).json()
    vals = response.values()
    keys = response.keys()
    vals = [round(el,3) for el in vals]
    response = dict(zip(keys,vals))
    print(f'Image: {i}, \n classification: {response}')