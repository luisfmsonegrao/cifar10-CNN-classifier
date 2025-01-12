import pickle
from flask import Flask
from flask import request
from flask import jsonify
import torch
from torchvision import transforms
import os,sys
import numpy as np

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

host_name = os.getenv("HOST_NAME")
if host_name == "REMOTE CONTAINER":
    device = torch.device("cpu")
    input_file = '/app/cifar_model_cpu_v1.bin'
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    input_file = root_path+'\\models\\cifar_model_v1.bin'

with open(input_file,'rb') as f_in:
    transform, model = pickle.load(f_in)

mean = [0.4914,0.4822,0.4465]
stdev = [0.2470,0.2435,0.2616]

transform = transforms.Normalize(mean,stdev)
model.eval()
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

softmax = torch.nn.Softmax(dim=1)
app = Flask('CIFAR')
@app.route('/predict',methods=['POST'])
def predict_endpoint():
    #print("model app")
    im = request.get_json()
    im = np.array(im)
    im = torch.tensor(im)
    im = im.to(device = device,dtype = torch.float)
    #im = transform(im)
    pred = model(im)
    pred = softmax(pred)
    pred_list = pred.tolist()
    pred_list = [round(el,3) for el in pred_list[0]]
    result = dict(zip(categories,pred_list))

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=9696)





