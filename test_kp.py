import torch
from lib.rcnn.keypoint_rcnn import *
from torchvision.models.detection import keypoint_rcnn as kp
from lib.ops.selector import ObjectSelector
from lib.ops.normalizer import KeypointNormalizer
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
import warnings
warnings.filterwarnings("ignore")
from random import uniform as U

num_images = 2
num_detections = 3
num_kp = 14
eval_ = False

path = 'data/val2017/000000471893.jpg'
img = Image.open(path).convert("RGB")
transform = ToTensor()
imgs = [transform(img) for _ in range(num_images)]
keypoints = torch.cat([torch.cat([Tensor([[10.*U(0,1), 10.*U(0,1), float(U(0,1)>.12)]]) for _ in range(num_kp)], dim=0).unsqueeze(0) for _ in range(num_detections)], dim=0)
print(keypoints.shape)
targets = [
    {
        'boxes': torch.cat([Tensor([[0., 0., 10., 10.]]) for _ in range(num_detections)], dim=0),
        'labels': (torch.randn(num_detections)>0.5).to(torch.int64),
        'image_id': torch.arange(num_detections)[i],
        'keypoints': keypoints,
        'area': U(100, 10000),
    } for i in range(num_images)
]

print('fm:')
print('---')
model = keypointrcnn_resnet50_fpn(num_classes=2)
if eval_:
    model.eval()
    features, detections = model(imgs)
else:
    features, detections, losses = model(imgs, targets)
    print('losses: ', losses)
print('features type', type(features), features[0].shape)
print('kp0', detections[0]['keypoints'].shape)
print('kp0 x_max', torch.max(detections[0]['keypoints'][:,0]))
print('bbx0', detections[0]['boxes'].shape)
print('lab0', detections[0]['labels'].shape)
print()

# print('torchvision:')
# print('------------')
# model = kp.keypointrcnn_resnet50_fpn(num_classes=2)
# model.eval()
# t_detections = model(imgs)
# print('kp0', t_detections[0]['keypoints'].shape)
# print('bbx0', t_detections[0]['boxes'].shape)

print('selection fm:')
print('-------------')
selector = ObjectSelector()
d, f = selector.select(detections, features)
print('shape fm selected: ',f[0].shape)
print('shape boxes selcted 0', d[0]['boxes'].shape)

print('kp normalizer:')
print('--------------')
normalizer = KeypointNormalizer([t['area'] for t in targets])
norm_d = normalizer.normalize(d)
inv_norm_d = normalizer.inverse_normalize(d)
print('kp00 bijective: ', torch.allclose(d[0]['keypoints'][0], inv_norm_d[0]['keypoints'][0]))
print('kp01 bijective: ', torch.allclose(d[0]['keypoints'][1], inv_norm_d[0]['keypoints'][1]))
print('kp10 bijective: ', torch.allclose(d[1]['keypoints'][0], inv_norm_d[1]['keypoints'][0]))
print('kp11 bijective: ', torch.allclose(d[1]['keypoints'][1], inv_norm_d[1]['keypoints'][1]))