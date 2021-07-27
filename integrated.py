import os
import cv2
import dlib
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import pretrainedmodels
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np
from PIL import Image

from adafruit_servokit import ServoKit
import board
import busio

import serial

print("All imported")

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x

preprocess = transforms.Compose([
   transforms.Resize(299),
   transforms.CenterCrop(299),
   transforms.ToTensor(),
   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict(frame, warm):
    height, width = frame.shape[:2]
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    image_tensor = preprocess(pil_im)
    with torch.no_grad():
        model.eval()
        x1 = Variable(image_tensor.unsqueeze(0))
        x1 = x1.to(device)
        y = model(x1)  # forward pass
        if warm:
            return
        index = y.data.cpu().numpy().argmax()
        print(y.data)
        print(index)
        kit.servo[14].angle=330*index
        print(['negative', 'positive'][index])
        if index == 0:
            ser.write(b'o')
            resp = ser.readline()
            time.sleep(5)
            rec = chr(int(resp.decode('utf-8').rstrip()))
            print(rec)
        else:
            ser.write(b'O')
            resp = ser.readline()
            time.sleep(5)
            rec = chr(int(resp.decode('utf-8').rstrip()))
            print(rec)
        ser.write(b'C')
        resp = ser.readline()
        rec = chr(int(resp.decode('utf-8').rstrip()))
        print(rec)


    return frame


def cutimg(image, warm):
    detector = dlib.get_frontal_face_detector()
    h, w = image.shape[:2]
    y1 = int(h/4*1)
    y2 = int(h/4*3)
    x1 = int(w/4*1)
    x2 = int(w/4*3)
    image = image[y1:y2, x1:x2]
    image = cv2.resize(image, dsize=(720, 640))
    rects = detector(image, 1)
    if warm:
        return 

    if len(rects) > 0:
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = rect_to_bb(rect)
            return image[y:y+h, x:x+w]
    else:
        return False 
        
def execute(change, warm=False):
    image = change['new']
    
    if not warm:
        print("Started cutting")
        start = time.time()
        image = cutimg(image, warm)
        end = time.time()
        print("Dlib time: {}".format(end-start))
    else:
        print("Warming dlib")
    if warm:
        predict(image, warm)
    if image is False:
        ser.write(b'w')
        resp = ser.readline()
        time.sleep(5)
        rec = chr(int(resp.decode('utf-8').rstrip()))
        print(rec)
        ser.write(b'C')
        resp = ser.readline()
        rec = chr(int(resp.decode('utf-8').rstrip()))
        print(rec)
        return
    
    print("Start prediction")
    start = time.time()
    print(image)
    predict(image, warm)
    end = time.time()
    if not warm:
        print(f"Prediction time: {end - start - 5}")
    else:
        print(f"Prediction time: {end - start}")


if __name__ == "__main__":
    print("Loading PyTorch")
    PATH = './models/2classes_epoch_100_acc_0.7849755035383779.pkl'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
         print("CUDA available")
    
    load_mdl_time1 = time.time()
    model_ft = pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained=None)
    num_ftrs = model_ft.last_linear.in_features
    model_ft.last_linear = nn.Linear(num_ftrs, 2)
    model_ft = nn.DataParallel(model_ft)
    model_ft.load_state_dict(torch.load(PATH))
    model_ft = model_ft.to(device)
    model = model_ft
    load_mdl_time2 = time.time()
    print(f'model loaded in {load_mdl_time2 - load_mdl_time1}')

    print("Initializing Servos")
    i2c_bus=(busio.I2C(board.SCL, board.SDA))
    print("Initializing serial communication")
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    rec = 1
    print("Warming LED")
    while 1:
        ser.write(b'C')
        try:
            resp = ser.readline()
            rec = chr(int(resp.decode('utf-8').rstrip()))
            print(rec)
        except:
            continue
        break

    print("Initilizing servo")
    kit = ServoKit(channels=16, address=96, i2c=i2c_bus)
    kit.servo[14].actuation_range = 360
    kit.servo[14].angle=0
    time.sleep(1.5)
    kit.servo[14].angle=360
    print("Initilizing camera")
    cap_time1 = time.time()
    cap = cv2.VideoCapture(0)
    print('Trying capture')
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    print("Capture no error") 
    cap_time2 = time.time()
    print(f"Capture loaded in {cap_time2 - cap_time1}s")
    print("Initializing PyTorch")
    ret, frame = cap.read()
    execute({'new': frame}, warm=True)
    print("Done Initialization!")



    while True:
        ret, frame = cap.read()
    
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    
        if cv2.waitKey(25) == ord('q'):
            break
        execute({'new': frame})
    
    cap.release()
    cv2.destroyAllWindows()