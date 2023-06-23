import numpy as np
import argparse
import torch
from src.transform import SSDTransformer
import cv2
from PIL import Image
import os

from src.utils import generate_dboxes, Encoder, colors, coco_classes
from src.model import SSD, ResNet

root_path = os.path.dirname(os.path.abspath(__file__))
cls_threshold = 0.3
nms_threshold = 0.5

class PlateDetection():
    def __init__(self):
        self.model = SSD(backbone=ResNet())
        checkpoint = torch.load(root_path+'/easyocr/ssd.pth',map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        dboxes = generate_dboxes()
        self.encoder = Encoder(dboxes)
        self.transformer = SSDTransformer(dboxes, (300, 300), val=True)
    

    def predict(self, img):
        output_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert('RGB')
        img, _, _, _ = self.transformer(img, None, torch.zeros(1,4), torch.zeros(1))
        number_plates = []
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            ploc, plabel = self.model(img.unsqueeze(dim=0))
            result = self.encoder.decode_batch(ploc, plabel, nms_threshold, 20)[0]
            loc, label, prob = [r.cpu().numpy() for r in result]
            best = np.argwhere(prob > cls_threshold).squeeze(axis=1)
            loc = loc[best]
            label = label[best]
            prob = prob[best]
            if len(label) > 0:
                if label[0] == 1:
                    height, width, _ = output_img.shape
                    loc[:, 0::2] *= width
                    loc[:, 1::2] *= height
                    loc = loc.astype(np.int32)
                    for box, lb, pr in zip(loc, label, prob):
                        category = coco_classes[lb]
                        color = colors[lb]
                        xmin, ymin, xmax, ymax = box
                        xmin, ymin, xmax, ymax = max(0,xmin), max(0,ymin), min(xmax,width-1), min(ymax,height-1)
                        p = output_img[ymin:ymax,xmin:xmax]
                        return self.predict(p)
                else:
                    if len(loc) > 0:
                        height, width, _ = output_img.shape
                        loc[:, 0::2] *= width
                        loc[:, 1::2] *= height
                        loc = loc.astype(np.int32)
                        for box, lb, pr in zip(loc, label, prob):
                            category = coco_classes[lb]
                            color = colors[lb]
                            xmin, ymin, xmax, ymax = box
                           
                            p = output_img[ymin:ymax,xmin:xmax]
                            number_plates.append(p)
                    else:
                        number_plates = []
        return number_plates
    
plate_detection_obj = PlateDetection()