import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join
import pickle as pk
from utility.iofile import Bbox_set
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL.ImageDraw import Draw
from PIL import Image
from model import *


class rectangle(object):
    def __init__(self,x=0,y=0,w=0,h=0):
        if isinstance(x,(tuple,list)):
            x,y,w,h=x
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.x2 = x+w
        self.y2 = y+h
        
    def area(self):
        return self.w*self.h if self.w>0 and self.h>0 else 0
    
    def perimeter(self):
        return 2*(self.w+self.h) if self.w>0 and self.h>0 else 0
    
    def overlap(self,rect):
        if isinstance(rect,(tuple,list)):
            rect = rectangle(*rect)
        x = max(self.x, rect.x)
        y = max(self.y, rect.y)
        w = min(self.x2, rect.x2) - x
        h = min(self.y2, rect.y2) - y
        return rectangle(x,y,w,h)
    
    def boundrect(self,rect):
        if isinstance(rect,(tuple,list)):
            rect = rectangle(*rect)
        x = min(self.x, rect.x)
        y = min(self.y, rect.y)
        w = max(self.x2, rect.x2) - x
        h = max(self.y2, rect.y2) - y
        return rectangle(x,y,w,h)
    
    def overlap_IoBB(self,rect):
        if isinstance(rect,(tuple,list)):
            rect = rectangle(*rect)
        return self.overlap(rect).area()/self.area()
    
    def overlap_IoU(self,rect):
        if isinstance(rect,(tuple,list)):
            rect = rectangle(*rect)
        return self.overlap(rect).area()/(self.area()+rect.area()-self.overlap(rect).area())
    
    def dist(self, x, y):
        if self.x<=x and x<=self.x2 and self.y<=y and y<=self.y2:
            return 0
        elif self.x<=x and x<=self.x2:
            return max(y-self.y2, self.y-y)
        elif self.y<=y and y<=self.y2:
            return max(x-self.x2, self.x-x)
        else:
            return np.sqrt(max(y-self.y2, self.y-y)**2+max(x-self.x2, self.x-x)**2)
        
    def __repr__(self):
        return "rectangle({},{},{},{})".format(self.x,self.y,self.w,self.h)
    
def returnCAM(feature_conv, weight_softmax, size, class_idx, bias=0):
    # generate the class activation maps upsample to size
    size_upsample = size if isinstance(size, tuple) else (size, size)
    bz, nc, h, w = feature_conv.shape
    if isinstance(bias, (int, float)):
        bias = [bias]*len(weight_softmax)
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))+bias[idx]
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam
    

if __name__=="__main__":    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    bboxset = Bbox_set( transform=transform)
    bbox_loader = DataLoader(bboxset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
    
    model_name = '/home/hddraid/Mao/ImageGCN/models/best_singlealex_relation_RGB_tr0.7_norm.pth.tar'
    relations = ['pid', 'age', 'gender', 'view']
    model = SingleLayerImageGCN(relations, encoder='singlealex', inchannel=3, share_encoder='partly')    
    checkpoint = torch.load(model_name)    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()    
    encoder = model.layer.encoder
    # weight = model.layer.gcn.classifier.weight.data.numpy()
    # bias = model.layer.gcn.classifier.bias.data.numpy()
    
    weight = model.layer.gcn['self'].linear.weight.data.numpy()
    bias = model.layer.gcn['self'].linear.bias.data.numpy()
    
    # model = MyAlexNet(14)
    # model_name = '/home/hddraid/Mao/ImageGCN/models/baselines/best_alex_s1_tr0.7.pth.tar'   
    # checkpoint = torch.load(model_name)    
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()    
    # encoder = model.features
    # weight = model.classifier.weight.data.numpy()
    # bias = model.classifier.bias.data.numpy()
    
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    encoder[-2][2].register_forward_hook(hook_feature)
    
    bbox = pd.read_csv(join(Bbox_set.path,'BBox_list_2017.csv'))
    bbox['rect'] = bbox.iloc[:,[2,3,4,5]].apply(lambda x: tuple(x),axis=1)
    bbox = bbox[['Image Index','Finding Label','rect']]
    bbox['detected'] = ''
    bbox['iou'] = ''
    bbox['iobb'] = ''
    with torch.no_grad():
        for i, data in enumerate(bbox_loader):            
            img_variable, label, name, orgrect, idx=data['image'], data['label'].numpy()[0],  \
                                                data['name'][0], data['bbox'].numpy()[0],  data['index'].numpy()[0]
            print('processing image '+str(idx)+':'+name)
            logit = encoder(img_variable)
            CAMs = returnCAM(features_blobs[0], weight, 224, [label], 0)
            features_blobs = []
            b = orgrect*256/1024-np.array([16,16,0,0], dtype=np.float32)
            heatmap = cv2.applyColorMap(CAMs[0],cv2.COLORMAP_JET)
            image = Image.open(join(bboxset.root_dir, name)).convert('RGB')        
            draw = Draw(image)

            pos = np.where(CAMs[0]==CAMs[0].max())
            ret,mask=cv2.threshold(CAMs[0], 180, 255, cv2.THRESH_BINARY)
            x,contour,hie=cv2.findContours(mask, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE )
            rect=set()
            for cnt in contour:
                x,y,w,h = cv2.boundingRect(cnt)
                for i,j in zip(*pos):
                    if rectangle(x,y,w,h).dist(j,i)==0:
                        rect.add((x,y,w,h))
                        heatmap=cv2.rectangle(heatmap,(x,y),(x+w,y+h),(0,0,255),1)
                        draw.rectangle(((x,y),(x+w,y+h)), outline=(255,0,0))
                        break 
            bbox.loc[idx,'detected']=rect
            bbox.loc[idx,'iou']=[rectangle(c).overlap_IoU(tuple(b)) for c in rect]
            bbox.loc[idx,'iobb']=[rectangle(c).overlap_IoBB(tuple(b)) for c in rect]

            
            cv2.rectangle(heatmap, tuple(b[:2]), tuple(b[:2]+b[2:4]), (255,0,0), 1)
            draw.rectangle(( tuple(b[:2]), tuple(b[:2]+b[2:4])), outline=(0,0,255))
            cv2.imwrite('/home/hddraid/Mao/ImageGCN/heatmaps/'+'label'+str(label)+'_'+name, heatmap)
            image.save('/home/hddraid/Mao/ImageGCN/bbox/'+'label'+str(label)+'_'+name)


    classes = {'Atelectasis':0, 'Cardiomegaly':1, 'Effusion':2, 'Infiltrate':3, \
                            'Mass':4, 'Nodule':5, 'Pneumonia':6, 'Pneumothorax':7, \
                            'Consolidation':8, 'Edema':9, 'Emphysema':10, 'Fibrosis':11, \
                            'Pleural_Thickening':12, 'Hernia':13   }

    res_iou=[]
    res_iobb=[]
    thres = np.linspace(0.1,0.9,9)
    for thr in thres:
        temp_iou=bbox['iou'].apply(lambda x: np.array(x)<thr)
        temp_iobb=bbox['iobb'].apply(lambda x: np.array(x)<thr)
        bbox['iou_match'+str(thr)]=temp_iou.apply(lambda x: not x.all())
        bbox['iou_fp'+str(thr)]=temp_iou.apply(lambda x: x.sum())
        bbox['iobb_match'+str(thr)]=temp_iobb.apply(lambda x: not x.all())
        bbox['iobb_fp'+str(thr)]=temp_iobb.apply(lambda x: x.sum())
        res_iou.append( bbox.groupby('Finding Label')['iou_match'+str(thr),'iou_fp'+str(thr)].mean())
        res_iobb.append( bbox.groupby('Finding Label')['iobb_match'+str(thr),'iobb_fp'+str(thr)].mean())
        
    pd.concat(res_iou, axis=1).to_csv('/home/hddraid/Mao/ImageGCN/bbox_res/res_iou.csv')
    pd.concat(res_iobb, axis=1).to_csv('/home/hddraid/Mao/ImageGCN/bbox_res/res_iobb.csv')
    bbox.to_csv('/home/hddraid/Mao/ImageGCN/bbox_res/bbox_detect.csv')


    