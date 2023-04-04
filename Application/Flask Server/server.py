from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models
import h5py
import faiss
import numpy as np
from netvlad import NetVLAD
from torchsummary import summary
import torch.nn.functional as F
from os.path import join

from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

@app.before_first_request
def setupModel():
    print("Setting up model")
    # Load model here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    encoder = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]
    for l in layers[:-5]: 
        for p in l.parameters():
            p.requires_grad = False
    encoder = nn.Sequential(*layers)
    net_vlad = NetVLAD(num_clusters=64, dim=512)
    encoder.add_module('pool', net_vlad)
    encoder.to(device)
    resume_ckpt = join('data/vgg16_checkpoint', 'checkpoints', 'model_best.pth.tar')
    checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint['state_dict'])
    print('Loaded model from {}'.format(resume_ckpt))

class SemanticSeg(Resource):
    def get(self):
        return {'hello': 'world'}

class getNearestNeighbours(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(SemanticSeg, '/segmentation')
api.add_resource(getNearestNeighbours, '/nearestNeighbours')

if __name__ == '__main__':
    app.run()