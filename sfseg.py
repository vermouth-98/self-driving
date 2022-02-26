import os
import logging
import time
import os
import logging
import time
import argparse
import datetime


# import apex
from PIL import Image
import numpy as np
import cv2,sys
from mss import mss
from pathlib import Path
import torch
from torch.backends import cudnn
import torchvision.transforms as transforms
from SFSegNets.network import get_net
from SFSegNets.optimizer import restore_snapshot
from SFSegNets.datasets import cityscapes
from SFSegNets.config import assert_and_infer_cfg

cudnn.benchmark = True
torch.cuda.empty_cache()
snapshot = "SFSegNets/res101_sfnet.pth"
arch = "SFSegNets.network.sfnet_resnet.DeepR101_SF_deeply" #network architecture used for inference'
device  = torch.device("cuda:0")
assert_and_infer_cfg(False,False,train_mode=False)

dataset_cls = cityscapes
net = get_net(arch,dataset_cls, criterion=None)
net = torch.nn.DataParallel(net).cuda()
print('Net built.')
net, _ = restore_snapshot(net, optimizer=None, snapshot=snapshot, restore_optimizer_bool=False)
net.half().eval()
# get data transforms.ToPILImage(),
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.Resize([512,512]),transforms.ToTensor(), transforms.Normalize(*mean_std)])
#=========================yolov5+deep_sort========================

# yolo_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize([640,640]),transforms.ToTensor()])
#=========================real-time===============================
start_time = time.time()
mon = {'top' : 0, 'left' : 0, 'width' : 1920, 'height' : 1080}
sct = mss()

monitor_1 = sct.monitors[1]
# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
font = cv2.FONT_HERSHEY_SIMPLEX
previous_time = 0
flag= True
dt, seen = [0.0, 0.0, 0.0, 0.0], 0
print("starting ...")
while True :
    
    frame = sct.grab(monitor_1)
    # frame = cv2.cvtColor(np.array(frame),cv2.COLOR_BGRA_RGB)
    frame = Image.frombytes("RGB", frame.size, frame.bgra, "raw", "BGRX")

    img_tensor = img_transform(frame).unsqueeze(0).cuda().half()
    
    with torch.no_grad():
        pred = net(x=img_tensor)
    
    pred = pred.cpu().numpy().squeeze()
    pred = np.argmax(pred, axis=0)
    colorized = dataset_cls.colorize_mask(pred)
    colorized = np.array(colorized.convert('RGB'))
    colorized = cv2.resize(colorized,frame.size)
    # overlap = cv2.addWeighted(np.array(frame), 0.5, colorized, 0.5, 0)
    #====================yolo========================
    txt1 = 'fps: %.1f' % ( 1./( time.time() - previous_time ))
    cv2.putText(colorized, txt1, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    previous_time = time.time()
    cv2.imshow('Computer Vision', colorized)

    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break

end_time = time.time()

# print('Results saved.')
# print('Inference takes %4.2f seconds, which is %4.2f seconds per image, including saving results.' % (end_time - start_time, (end_time - start_time)/len(images)))
