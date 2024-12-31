from ts.torch_handler.image_classifier import ImageClassifier
from ts.utils.util import map_class_to_label
from torchvision import transforms
import numpy as np
import json
import torch

class CIFAR10Handler(ImageClassifier):
    topk=10
    image_processing = transforms.Compose(
        [
            transforms.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2470,0.2435,0.2616]),
        ]
    )

    def preprocess(self, data):
        input_json = data[0].get("body",None)
        if input_json is None:
            raise ValueError("Input data is empty")
        input_data = json.loads(input_json)
        input_array = np.array(input_data)
        input_tensor = torch.tensor(input_array)
        input_tensor = input_tensor.to(device='cpu',dtype=torch.float)
        input_tensor = self.image_processing(input_tensor)
        return input_tensor
    

