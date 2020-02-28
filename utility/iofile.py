import pandas as pd
import numpy as np
import pickle as pk
import copy
import gc
from torch.utils.data import Dataset, DataLoader
from os.path import join
from PIL import Image
from .preprocessing import adj_from_series
from .selfdefine import FlexCounter
from heapq import nlargest
from collections import Counter




def save_obj(obj, name ):
    with open(name , 'wb') as f:
        pk.dump(obj, f, pk.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name , 'rb') as f:
        return pk.load(f)
    

class ChestXray_Dataset(Dataset):
    """ChestXray dataset."""
    
    def __init__(self, path='/home/hddraid/shared_data/chest_xray8/', mode='RGB', adjgroup=True, neib_samp='relation', \
                 relations= ['pid', 'age', 'gender', 'view'], k = 16, graph_nodes='current',  transform=None):
        """
        Args:
            csv_labelfile (string): Path to the csv file with labels.
            csv_bboxfile (string): Path to the csv file with bbox.
            root_dir (string): Directory with all the images.
            use (string): 'train' or 'validation' or 'test'
            transform (callable, optional): Optional transform to be applied on a sample.
            split(string): train,val,test split. ('specified', 'random', 'bboxrandom')
            mode(string): convert to a image mode ('L', 'RGB')  
        """
        self.path=path
        csv_labelfile=join(path, 'Data_Entry_2017.csv')
        csv_bboxfile=join(path,'BBox_list_2017.csv')
        root_dir=join(path,'images/cropedimages')
        self.all_label_df = pd.read_csv(csv_labelfile, usecols=range(7))
        self.all_label_df.columns= ['name','label','followup', 'pid', 'age', 'gender', 'view']               
        self.label_df = self.all_label_df
        self.bbox = pd.read_csv(csv_bboxfile, usecols=range(6)) 
        self.bbox.columns = ['name','label','x', 'y', 'w', 'h']
        self.bbox_label_df=self.all_label_df.loc[self.all_label_df['name'].isin(self.bbox['name'])]
        self.root_dir = root_dir
        self.mode = mode
        self.adjgroup = adjgroup
        self.neib_samp = neib_samp
        self.k=k
        self.gnode = graph_nodes
        self.classes = {'Atelectasis':0, 'Cardiomegaly':1, 'Effusion':2, 'Infiltration':3, \
                        'Mass':4, 'Nodule':5, 'Pneumonia':6, 'Pneumothorax':7, \
                        'Consolidation':8, 'Edema':9, 'Emphysema':10, 'Fibrosis':11, \
                        'Pleural_Thickening':12, 'Hernia':13   }
        self.transform = transform
        self.relations = relations
        self.all_grp = self.creat_adj(self.label_df ) 
        self.grp = self.all_grp

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        sample = self._getimage(idx)
        if self.neib_samp == 'relation':
            img = self.label_df.iloc[idx]
            impt = self.impt_sample(img, k=1) 
            sample['impt'] = impt
        return sample
    
    def _getimage(self, idx, byindex=False, level=0):
        img = self.all_label_df.loc[idx] if byindex else self.label_df.iloc[idx]
        image = Image.open(join(self.root_dir, img[0])).convert(self.mode)
        labels = np.zeros(len(self.classes),dtype=np.float32)
        labels[[self.classes[x.strip()] for x in img[1].split('|') if x.strip() in self.classes]] = 1          
        
        sample = {'image': image, 'label': labels, 'pid':img[3],  'age':img[4], 'gender':img[5], 'view':img[6],  \
                  'name': img[0], 'index':img.name}
        if level==0:
            sample['dataset'] = self
            if self.neib_samp in ('sampling', 'best'):
                w = sum([(FlexCounter(grp[img[r]])/len(grp[img[r]]) if img[r] in  grp else FlexCounter())   \
                         for r, grp in self.tr_grp.items()], Counter())
                sample['weight'] = w           

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
    
    def impt_sample(self, img, method='relation', k=1, base='train'):
        """
        sampling the important k samples for img.
        method: "sample"--random choose by probability
                "best"--choose the most important
                "relation"--random choose k for each relation
        base: choose the basic set, "train" or "all"
        """
        if base == "train":
            grps = self.tr_grp
        elif base == "all":
            grps = self.all_grp 
            
        if method=='relation':
            impt_sample=[]
            for r, grp in grps.items():
                if img[r] in grp:
                    neibs = grp[img[r]].drop(img.name, errors = 'ignore')
                    if not neibs.empty:
                        impt_sample += np.random.choice(neibs, k, replace=False).tolist()
            return impt_sample    
    
        w = sum([FlexCounter(grp[img[r]])/len(grp[img[r]]) for r, grp in grps.items()], Counter())
        w.pop(img.name, None)
        if method == "sample":   
            p = FlexCounter(w)/sum(w.values())
            impt_sample = np.random.choice(list(p.keys()), k, replace=False, p=list(p.values()))
        elif method == 'best':
            impt_sample = nlargest(k, w, key = w.get) 
            
        return impt_sample
    
    def creat_adj(self, label_df, adjgroup=True ):
        if self.gnode=='current':
            adj = {r:adj_from_series(label_df[r], groups= adjgroup) for r in self.relations}
            # pat_adj = adj_from_series(label_df['pid'], groups= adjgroup)
            # gen_adj = adj_from_series(label_df['gender'], groups= adjgroup)
            # age_adj = adj_from_series(label_df['age'], groups= adjgroup)
            # view_adj= adj_from_series(label_df['view'], groups= adjgroup)
        else:
            pass
        
        # adj = {'pid':pat_adj, 'gender': gen_adj, 'age': age_adj, 'view':view_adj}
        return adj
            
    def tr_val_te_split(self,  split='random', tr_pct=0.7 ):
        n_all=len(self.all_label_df) 
        np.random.seed(0)
        if split=='specified':
            te = pd.read_csv(join(self.path,'test_list.txt'),header=None)[0]
            tr_val = pd.read_csv(join(self.path,'train_val_list.txt'),header=None)[0]
            tr, val = np.split(tr_val.sample(frac=1, random_state=0),[int(len(tr_val)*0.875),])
            tr_df = self.all_label_df.loc[self.all_label_df['name'].isin(tr)]
            val_df = self.all_label_df.loc[self.all_label_df['name'].isin(val)]
            te_df = self.all_label_df.loc[self.all_label_df['name'].isin(te)]
        elif split == 'random':
            tr_df, val_df, te_df = np.split(self.all_label_df.sample(frac=1, random_state=0),[int(n_all*0.7), int(n_all*0.8),])
            tr_df = tr_df.sample(n=int(n_all*tr_pct),random_state=0)
        elif split == 'bboxrandom':
            bbox_idx = self.bbox['name'].drop_duplicates()
            n_bbox = len(bbox_idx)
            nobox_idx =self.all_label_df['name'][ ~self.all_label_df['name'].isin(bbox_idx)]
            n_nobox = len(nobox_idx)
            tr_box, val_box, te_box = np.split(bbox_idx.sample(frac=1, random_state=0),[int(n_bbox*0.7), int(n_bbox*0.8),])
            tr_nobox,val_nobox,te_nobox=np.split(nobox_idx.sample(frac=1, random_state=0),[int(n_nobox*0.7), int(n_nobox*0.8),])
            tr, val, te = tr_box.append(tr_nobox), val_box.append(val_nobox), te_box.append(te_nobox)
            tr_df = self.all_label_df.loc[self.all_label_df['name'].isin(tr)]
            val_df = self.all_label_df.loc[self.all_label_df['name'].isin(val)]
            te_df = self.all_label_df.loc[self.all_label_df['name'].isin(te)]
        else:
            raise Error('split %s is not defined'%(split))
        
        self.tr_grp = self.creat_adj(tr_df)
        
        tr_set, val_set, te_set = copy.copy(self), copy.copy(self), copy.copy(self)
        tr_set.label_df, val_set.label_df, te_set.label_df  = tr_df, val_df, te_df
        # print('creating groups and importances for training set......')
        # if self.adjgroup:
        #     tr_set.grp = tr_set.creat_adj(tr_set.label_df)
        # else:
        #     adj = tr_set.creat_adj(self.all_label_df,False)
        #     impt = pd.DataFrame(adj['age']+adj['pid']+adj['gender']+adj['view'],  \
        #                              index = self.all_label_df.index, columns = self.all_label_df.index)
        #     del adj
        #     gc.collect()
        #     self.impt = impt[tr_df.index]

        
        return tr_set, val_set, te_set
        

class Bbox_set(Dataset):
    """chest x-ray boboxe test dataset."""

    classes = {'Atelectasis':0, 'Cardiomegaly':1, 'Effusion':2, 'Infiltrate':3, \
                        'Mass':4, 'Nodule':5, 'Pneumonia':6, 'Pneumothorax':7, \
                        'Consolidation':8, 'Edema':9, 'Emphysema':10, 'Fibrosis':11, \
                        'Pleural_Thickening':12, 'Hernia':13   }
    path='/home/hddraid/shared_data/chest_xray8/'
    
    def __init__(self, csv_bboxfile=join(path,'BBox_list_2017.csv'), \
                 root_dir=join(path,'images/cropedimages'), transform=None):
        """
        Args:
            csv_bboxfile (string): Path to the csv file with bbox.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.bbox=pd.read_csv(csv_bboxfile)             
        self.root_dir = root_dir        
        self.transform = transform

    def __len__(self):
        return len(self.bbox)

    def __getitem__(self, idx):
        img = self.bbox.loc[idx]
        img_name = img[0]
        image = Image.open(join(self.root_dir, img_name)).convert('RGB')
        label = self.classes[img[1]]
        
        bbox = img[[2,3,4,5]].values
        bbox = bbox.astype(np.float32)
        
        sample = {'image': image, 'label': label, 'name': img_name, 'bbox': bbox, 'index':idx}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample        
            
            
            
            
            
            
            