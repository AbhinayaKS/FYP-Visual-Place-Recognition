from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models
from io import BytesIO
import faiss
import numpy as np
from netvlad import NetVLAD
from torch.autograd import Variable
import torch.nn.functional as F
from os.path import join
import os
import torchvision.transforms as transforms
from PIL import Image
from werkzeug.serving import WSGIRequestHandler

from flask import Flask, request, send_file, make_response
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        normalize,
])

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

@app.before_first_request
def setupModel():
    print("Setting up model")
    torch.manual_seed(42)

    encoder = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    layers = list(encoder.features.children())[:-2]
    for l in layers[:-5]: 
        for p in l.parameters():
            p.requires_grad = False
    encoder = nn.Sequential(*layers)
    global model
    model = nn.Module()
    model.add_module('encoder', encoder)
    net_vlad = NetVLAD(num_clusters=64, dim=512)
    model.add_module('pool', net_vlad)

    resume_ckpt = join('data/vgg16_checkpoint', 'checkpoints', 'model_best.pth.tar')
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    print('Loaded model from {}'.format(resume_ckpt))

class getNearestNeighbours(Resource):
    def get(self):

        with torch.no_grad():
            descriptors = torch.empty(10, 32768).to(device)
            query = torch.empty(1, 32768).to(device)
            count = 0
            if 'image' in request.args:
                image = request.args['image']
                img = Image.open('data/' + image)
                img = transform_pipeline(img)
                img = img.unsqueeze(0)  
                img = Variable(img)
                img = img.to(device)
                image_encoding = model.encoder(img)
                vlad_encoding = model.pool(image_encoding)
                query[count] = vlad_encoding
            else:
                return {'error': 'No image specified'}
            
            for file in os.listdir('data/'):
                if file.endswith(".png"):
                    if file == 'primeArea.png':
                        continue
                    img = Image.open('data/' + file)
                    img = transform_pipeline(img)
                    img = img.unsqueeze(0)  
                    img = Variable(img)
                    img = img.to(device)
                    image_encoding = model.encoder(img)
                    vlad_encoding = model.pool(image_encoding)
                    descriptors[count] = vlad_encoding
                    count += 1
            faiss_index = faiss.IndexFlatL2(32768)
            faiss_index.add(descriptors.cpu().numpy())
            _, indices = faiss_index.search(query.cpu().numpy(), 1)
            count = 0
            for file in os.listdir('data/'):
                if file.endswith('.png'):
                    if count in indices:
                        byte_io = BytesIO()
                        with open('data/primeArea.png', 'rb') as f:
                            byte_io.write(f.read())
                            byte_io.seek(0)
                        response = make_response(send_file(byte_io, mimetype='image/png'))
                        response.headers['Connection'] = 'Keep-Alive'

                        return response
                    count += 1

        return {'neighbours': 'no neighbours found'}

class SemanticSeg(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(SemanticSeg, '/segmentation')
api.add_resource(getNearestNeighbours, '/nearestNeighbours')

if __name__ == '__main__':
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(host = '0.0.0.0', port = 5000)