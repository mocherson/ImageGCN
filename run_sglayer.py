

import argparse
from model import *
from utility.iofile import *
from utility.selfdefine import *
from utility.preprocessing import sparse_to_tensor
from utility.collate import mycollate
from torchvision import transforms
from torch import  optim
import time
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser(description='GCN for chestXray')
parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size for training (default: 16)')
parser.add_argument('--path', default='/share/fsmresfiles/ChestXRay/', type=str, help='data path')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
parser.add_argument('--gpu', type=int, default=-1, metavar='N', help='the GPU number (default auto schedule)')
parser.add_argument('-e','--encoder', default='alex', type=str, help='the encoder')
parser.add_argument('-r','--relations', default='all', type=str, help='the considered relations, pid, age, gender, view')
parser.add_argument('--pps', default='partly', type=str, help='the parameter sharing method (default partly)')
parser.add_argument('--use', default='train', type=str, help='train or test (default train)')
parser.add_argument('-m','--mode', default='RGB', type=str, help='the mode of the image')
parser.add_argument('-s','--neibor', default='relation', type=str, help='the neighbor sampling method (default: relation)')
parser.add_argument('--k',type=int, default=16,  metavar='N', help='the number of neighbors sampling (default: 16)')
parser.add_argument('-p','--train-percent',type=float, default=0.7,  metavar='N', help='the percentage of training data (default: 0.7)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',  help='manual epoch number (useful on restarts)')
parser.add_argument('-d','--weight_decay',type=float, default=0,  metavar='N', help='the percentage of training data (default: 0)')
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
batch_size = args.batch_size
enc = 'single'+args.encoder
neib = args.neibor
k = args.k
inchannel = 3 if args.mode=='RGB' else 1
mode = args.mode
tr_pct = args.train_percent
pps = args.pps
use = args.use
wd=args.weight_decay
relations = ['pid', 'age', 'gender', 'view'] if args.relations=='all' else [] if args.relations=='no' else [args.relations]

transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) ])

dataset = ChestXray_Dataset(path=args.path, mode=mode, neib_samp=neib, relations=relations, k=k, transform=transform)
train_set, validation_set, test_set = dataset.tr_val_te_split(tr_pct=tr_pct)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn= mycollate, pin_memory=True, num_workers=16)
validation_loader = DataLoader(validation_set, batch_size=batch_size, collate_fn= mycollate, shuffle=True, pin_memory=True, num_workers=16)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn= mycollate, pin_memory=True, num_workers=16)



# if enc=='alex':
#     encoder = MyAlexNet(14).features[:-2]
#     hc = 256
# elif enc=='res50':
#     encoder = MyResNet50(14).features[:-2]
#     hc = 2048
# elif enc=='vgg16bn':
#     encoder = MyVggNet16_bn(14).features[:-2]
#     hc = 512
# elif enc=='vgg16':
#     encoder = MyVggNet16(14).features[:-2]
#     hc = 512
# elif enc=='dens161':
#     encoder=MyDensNet161(14).features[:-2]
#     hc = 2208
# elif enc=='dens201':
#     encoder=MyDensNet201(14).features[:-2]
#     hc = 1920
# elif enc=='dens121':
#     encoder=MyDensNet121(14).features[:-2]
#     hc = 1024

if args.gpu>=0:
    torch.cuda.set_device(args.gpu)

model = SingleLayerImageGCN(relations, encoder=enc, inchannel=inchannel, share_encoder=pps).cuda()
# model =  nn.DataParallel(model)

criterion = W_BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),  lr=1e-5, amsgrad =False, weight_decay=wd,)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=4)

def train(train_loader, validation_loader, test_loader,  model, criterion, optimizer, iter_size=100):
    print('training.....')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    res=pd.DataFrame(columns=['epoch','iter','loss_tr','loss_val','avgroc_val','avgroc_test','Atelectasis', 'Cardiomegaly', 'Effusion', \
                              'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', \
                              'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'])

    # switch to train mode
    end = time.time()
    for epoch in range(args.epochs): 
        for i, data in enumerate(train_loader):
            # measure data loading time  
            model.train()

            inputs, targets, adj, k = data['image'].cuda(), data['label'].cuda(), data['adj'], data['k']

            # adj_mats1 = {key: sparse_to_tensor(value).cuda() for key, value in adj.items()}
            adj_mats2 = {key: sparse_to_tensor(value).to_dense()[k:].cuda() for key, value in adj.items()}
            data_time.update(time.time() - end)

            output = model(inputs,  adj_mats2,  k=k)

            loss = criterion(output, targets[k:])
            losses.update(loss.item(), targets[k:].size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % iter_size == 0 or i == len(train_loader):
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses))

                loss_val, avgroc_val, _, _ = validate(validation_loader, model, criterion)
                loss_test, avgroc_test, roc_test, pred = validate(test_loader, model, criterion) 
                scheduler.step(avgroc_val)
                
                res.loc[len(res)]=[epoch, i, losses.avg,loss_val,avgroc_val,avgroc_test]+list(roc_test)
                res.to_csv('./results/CXR14/%s/%s_%s_%s_tr%s_%s_wd%s.csv'%(args.relations, enc,neib, mode, tr_pct,pps,wd) )
                
                if loss_val<=res['loss_val'].min():
                    torch.save({'epoch':epoch, 'i':i, 'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()},
                         './models/CXR14/%s/checkpoint_%s_%s_%s_tr%s_%s_wd%s_bestloss.pth.tar'  \
                              %(args.relations, enc,neib,mode,tr_pct,pps,wd))
                    save_obj(pred, './results/CXR14/%s/%s_%s_%s_tr%s_%s_wd%s_bestloss.pkl'  \
                                       %(args.relations, enc,neib, mode,tr_pct,pps,wd))
                    
                if avgroc_val>=res['avgroc_val'].max():
                    torch.save({'epoch':epoch, 'i':i, 'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()},
                         './models/CXR14/%s/checkpoint_%s_%s_%s_tr%s_%s_wd%s_bestroc.pth.tar'  \
                              %(args.relations, enc,neib,mode,tr_pct,pps,wd))
                    save_obj(pred, './results/CXR14/%s/%s_%s_%s_tr%s_%s_wd%s_bestroc.pkl'  \
                                       %(args.relations, enc,neib, mode,tr_pct,pps,wd))
                    
                if avgroc_test>=res['avgroc_test'].max():
                    torch.save({'epoch':epoch, 'i':i, 'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()},
                         './models/CXR14/%s/checkpoint_%s_%s_%s_tr%s_%s_wd%s_bestroc_te.pth.tar'  \
                              %(args.relations, enc,neib,mode,tr_pct,pps,wd))
                    save_obj(pred, './results/CXR14/%s/%s_%s_%s_tr%s_%s_wd%s_bestroc_te.pkl'  \
                                       %(args.relations, enc,neib, mode,tr_pct,pps,wd))
                    
                
        
    return res

def validate(val_loader, model, criterion):
    print('Validation......')
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        outputs=[]
        labels=[]
        index=[]
        for i, data in enumerate(val_loader):
            inputs, targets, adj, k = data['image'].cuda(), data['label'].cuda(), data['adj'], data['k']
            # adj_mats1 = {key: sparse_to_tensor(value).cuda() for key, value in adj.items()}
            adj_mats2 = {key: sparse_to_tensor(value).to_dense()[k:].cuda() for key, value in adj.items()}

            output = model(inputs,  adj_mats2, k=k)
            loss = criterion(output, targets[k:])  
            losses.update(loss.item(),targets[k:].size(0))
            outputs.append(output)
            labels.append(targets[k:])
            index.append(data['index'][k:]) 

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % 20 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses))
        outputs =  F.sigmoid(torch.cat(outputs)).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()
        idx = torch.cat(index).cpu().numpy()
        
    roc = roc_auc_score(labels, outputs, average=None)
    avgroc = roc.mean()
    print('validate roc',roc)
    print('validate average roc',avgroc)
            
    return losses.avg, avgroc, roc, (idx, labels, outputs)

if use=='train':                
    res=train(train_loader, validation_loader, test_loader, model, criterion, optimizer, 1000)
    res.to_csv('./results/CXR14/%s/%s_%s_%s_tr%s_%s_wd%s.csv'%(args.relations, enc,neib, mode, tr_pct,pps,wd))    
elif use=='test':
    cp = torch.load('./models/CXR14/%s/checkpoint_%s_%s_%s_tr%s_%s_wd%s_bestroc.pth.tar'  \
                              %(args.relations, enc,neib,mode,tr_pct,pps,wd))
    model.load_state_dict(cp['state_dict'])
    loss_test, avgroc_test, roc_test, pred = validate(test_loader, model, criterion)
    save_obj(pred,  './results/CXR14/%s/pred_%s_%s.pkl'%(args.relations,enc,pps))

    
    
    
    
    
    
    
    
    
    
    
    

