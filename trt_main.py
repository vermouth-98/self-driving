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
from yolov5.utils.augmentations import letterbox
from SFSegNets.network import get_net
from SFSegNets.optimizer import restore_snapshot
from SFSegNets.datasets import cityscapes
from SFSegNets.config import assert_and_infer_cfg
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
half = False
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
cfg = get_config()
cfg.merge_from_file('deep_sort/configs/deep_sort.yaml')
deep_sort_model='osnet_x0_25'
deepsort = DeepSort(deep_sort_model,
                    device,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    )
yolo_model= f'{ROOT}/yolov5s.engine'
device = select_device(device)
model = DetectMultiBackend(yolo_model, device=device, dnn=False,data = "yolov5/data/coco128.yaml")
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz=(640, 640)
imgsz = check_img_size(imgsz, s=stride)
bs = 1
# names = model.module.names if hasattr(model, 'module') else model.names
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=True)  # warmup
# yolo_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize([640,640]),transforms.ToTensor()])
#=========================real-time===============================
start_time = time.time()
mon = {'top' : 0, 'left' : 0, 'width' : 1920, 'height' : 1080}
sct = mss()

monitor_1 = sct.monitors[1]
half &= (pt or jit or onnx or engine) and device.type != 'cpu'
# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
font = cv2.FONT_HERSHEY_SIMPLEX
previous_time = 0
flag= True
frame_id = 0
dt, seen = [0.0, 0.0, 0.0, 0.0], 0
print("starting ...")
while True :
    
    frame = sct.grab(monitor_1)
    # frame = cv2.cvtColor(np.array(frame),cv2.COLOR_BGRA_RGB)
    frame = Image.frombytes("RGB", frame.size, frame.bgra, "raw", "BGRX")
    # frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im0 = np.array(frame)
    # im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    # im0 = cv2.resize(im0,(512,512))
    # cv2.imwrite("hihi.jpg",im0)
    img_tensor = img_transform(frame).unsqueeze(0).cuda().half()
    
    with torch.no_grad():
        pred = net(x=img_tensor)
    
    pred = pred.cpu().numpy().squeeze()
    pred = np.argmax(pred, axis=0)
    colorized = dataset_cls.colorize_mask(pred)
    colorized = np.array(colorized.convert('RGB'))
    colorized = cv2.resize(colorized,frame.size)
    overlap = cv2.addWeighted(np.array(frame), 0.5, colorized, 0.5, 0)
    #====================yolo========================
    img = letterbox(im0,stride = stride,auto = pt)[0]
        # Convert
    
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    # img = yolo_transform(im0).unsqueeze(0).to(device)
    # expand for batch dim
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255
    if len(img.shape) == 3:
        img = img[None]
    # print(img.shape)
    
    pred = model(img,augment=False)
    # print(pred)
    pred = non_max_suppression(pred,0.25,0.45, None, False, max_det=1000)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    det = pred[0]
    
    seen += 1
    annotator = Annotator(overlap, line_width=2, pil=not ascii)
    if det is not None and len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        xywhs = xyxy2xywh(det[:, 0:4].round())
        confs = det[:, 4]
        clss = det[:, 5]
        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(),im0)
                # draw boxes for visualization
        if len(outputs) > 0:
            for j, (output, conf) in enumerate(zip(outputs, confs)):

                bboxes = output[0:4]
                id = output[4]
                cls = output[5]

                c = int(cls)  # integer class
                label = f'{id} {names[c]} {conf:.2f}'
                annotator.box_label(bboxes, label, color=colors(c, True))
    else:
        deepsort.increment_ages()
    overlap = annotator.result()
    #=========================================
    txt1 = 'fps: %.1f' % ( 1./( time.time() - previous_time ))
    cv2.putText(overlap, txt1, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    previous_time = time.time()
    cv2.imshow('Computer Vision', overlap)
    
    # print(txt1)
    frame_id+=1
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break

end_time = time.time()

# print('Results saved.')
# print('Inference takes %4.2f seconds, which is %4.2f seconds per image, including saving results.' % (end_time - start_time, (end_time - start_time)/len(images)))
