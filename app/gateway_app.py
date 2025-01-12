from PIL import Image
import requests
import io
import os, sys
from torchvision import transforms
import torch
import numpy as np
from flask import Flask
from flask import request, jsonify
import json

model_url = os.getenv("MODEL_SERVING_HOST","http//localhost:9696/predict")
model_url = model_url
cifar_mean = [0.4914,0.4822,0.4465]
cifar_stdev = [0.2470,0.2435,0.2616]
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

app = Flask("gateway")
@app.route('/predict',methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    image_data = preprocess_image(url)
    result = predict(image_data)
    return result

def preprocess_image(url):
    im = requests.get(url)
    pim = Image.open(io.BytesIO(im.content))
    tf =transforms.Compose([transforms.ToTensor(),transforms.Normalize(cifar_mean,cifar_stdev)])
    tim = tf(pim)
    tim = tim.unsqueeze(0)
    data = np.array(tim).tolist()
    return data

def predict(im):
    response = requests.post(model_url,data=json.dumps(im)).json()
    return response




