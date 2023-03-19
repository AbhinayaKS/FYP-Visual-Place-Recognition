import torchvision.transforms as transforms
import torch.nn as nn
from utils import countless
import numpy as np
import cv2
import torch


import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.samplers.distributed_sampler import defaultdict
from Mask2Former.mask2former import add_maskformer2_config

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

class SemanticLabels(nn.Module):
    '''Extract the semantic labels as a pooling layers'''
    def __init__(self):
        super(SemanticLabels, self).__init__()
        setup_logger()
        setup_logger(name="mask2former")
        self.cfg = get_cfg()
        add_deeplab_config(self.cfg)
        add_maskformer2_config(self.cfg)
        self.cfg.merge_from_file("Mask2Former/configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml")
        self.cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_swin_large_IN21k_384_bs16_90k/model_final_17c1ee.pkl'
        self.cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
        self.predictor = DefaultPredictor(self.cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        inputInTensor = input.squeeze(0)
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
        inputToGenerateLabels = invTrans(inputInTensor)
        transformPIL = transforms.ToPILImage()
        labelsBatch = torch.zeros(inputInTensor.shape[0],30, 40)
        for i, image in enumerate(inputToGenerateLabels):
        	inputInPILImage = transformPIL(image)
        	inputToGenerateLabels = cv2.cvtColor(np.array(inputInPILImage), cv2.COLOR_RGB2BGR)
        	semanticOutputs = self.predictor(inputToGenerateLabels)
        	labels = semanticOutputs['sem_seg'].argmax(0)
        	while labels.shape > (30,40):
            		labels = countless(labels)
        	labelsBatch[i] = labels
        labelsBatch.to(self.device)
        return labelsBatch
