import os
import sqlite3
import json
import torch
from torchvision.io import read_image
from torch.utils.data.dataset import Dataset
from PIL import Image

class OpenlprDataset(Dataset):
    def __init__(self, dbfile, img_dir, transform=None, target_transform=None):
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        #todo get unique from db
        self.label_map = {
            'vehicle': 1,
            'plate': 2
        }

        try:
            conn = sqlite3.connect(dbfile)
            cur = conn.cursor()

            selQuery = "SELECT filename,imheight,imwidth,isreviewed,lastreviewedat,isdeleted,needscropping,isbackground,imgareas FROM annotations"
            cur.execute(selQuery)
            rows = cur.fetchall()

            cur.close()
            conn.close()

            #optimize
            #convert the tuples into lists and decode the imgareas into an object
            self.rows = []
            for r in rows:
                row = list(r)
                # row[0] = row[0].replace('.jpg','')
                row[8] = json.loads(row[8])
                self.rows.append(row)


        except:
            print("Error opening and reading database [{}]".format(dbfile))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.rows[idx][0])
        # image = read_image(img_path)
        image = Image.open(img_path).convert("RGB")

        width, height = self.rows[idx][2],self.rows[idx][1]

        boxes = []
        labels = []
        for annotation in self.rows[idx][8]:
            labels.append(int(annotation['lblid']))
            boxes.append(
                [
                    int(annotation['x'])/width, 
                    int(annotation['y'])/height,
                    (int(annotation['x']) + int(annotation['width']))/width,
                    (int(annotation['y']) + int(annotation['height']))/height
                ]
            )
        
        labels = torch.tensor(labels)
        boxes = torch.tensor(boxes)
        
        if self.transform:
            # image = self.transform(image)
            image, (height, width), boxes, labels = self.transform(image, (height, width), boxes, labels)
        # if self.target_transform:
        #     label = self.target_transform(label)
        
        img_id = self.rows[idx][0]
        img_id = img_id[ 0:img_id.index(".") ]
        return image, img_id, (self.rows[idx][1], self.rows[idx][2]), boxes, labels
