import pickle
from flask import Flask
from flask import request
from flask import jsonify
from torchvision import transforms
from torch import cuda
from torch import tensor
from torch import float as tfloat
import os,sys
import numpy as np

file_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(file_path)
print(file_path)

host_name = os.getenv("HOST_NAME")

if host_name == "REMOTE CONTAINER":
    input_file = file_path+'\\cifar_model_v1.bin'
else:
    input_file = file_path+'\\models\\cifar_model_v1.bin'
    
with open(input_file,'rb') as f_in:
    transform, model = pickle.load(f_in)

mean = [0.4914,0.4822,0.4465]
stdev = [0.2470,0.2435,0.2616]

transform = transforms.Normalize(mean,stdev)
model.eval()
device = 'cuda' if cuda.is_available() else 'cpu'
#device = 'cpu'
model.to(device)
app = Flask('CIFAR')
categories = ['airplane',
 'automobile',
 'bird',
 'cat',
 'deer',
 'dog',
 'frog',
 'horse',
 'ship',
 'truck']

@app.route('/predict',methods=['POST'])
def predict_endpoint():
    im = request.get_json()
    im = np.array(im)
    im = tensor(im)
    im = im.to(device = device,dtype = tfloat)
    im = transform(im)
    pred = model(im)
    pred_list = pred.tolist()
    pred_list = [round(el,3) for el in pred_list[0]]
    result = dict(zip(categories,pred_list))
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=9696)





