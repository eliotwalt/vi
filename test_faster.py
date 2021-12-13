import torch
from lib.rcnn.faster_rcnn import *
from torchvision.models.detection import faster_rcnn as fp
from lib.ops.selector import ObjectSelector
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
import warnings
warnings.filterwarnings("ignore")

path = 'data/val2017/000000471893.jpg'
img = Image.open(path).convert("RGB")
transform = ToTensor()
imgs = [transform(img)]

print('fm:')
print('---')
model = fasterrcnn_resnet50_fpn(num_classes=2)
model.eval()
features, detections = model(imgs)
print('f0', features[0].shape)
print('bbx0', detections[0]['boxes'].shape)
print('lab0', detections[0]['labels'].shape)
print()

print('torchvision:')
print('------------')
model = fp.fasterrcnn_resnet50_fpn(num_classes=2)
model.eval()
t_detections = model(imgs)
print('bbx0', t_detections[0]['boxes'].shape)
print('lab0', t_detections[0]['labels'].shape)


print('selection fm:')
print('-------------')
selector = ObjectSelector()
d, f = selector.select(detections, features)
print('shape fm selected: ',f[0].shape)
print('shape boxes selcted 0', d[0]['boxes'].shape)