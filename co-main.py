
import argparse
import os
import random
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.utils as utils
import torch.nn.functional as F
from models import G_CIFAR10,G_MNIST,D_CIFAR10,D_MNIST,DPN92,ResNet18,glj_MNIST,LeNet5
from tensorboard_logger import configure, log_value

from data import generate_c_data, CIFAR10_Complementary

from train_test import test_acc,train_c



from torchvision import datasets, transforms

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enbaled = True



def args():
    FLAG = argparse.ArgumentParser(description='ACGAN Implement With Pytorch.')
    FLAG.add_argument('--dataset', default='MNIST', help='CIFAR10 | MNIST')
    FLAG.add_argument('--savingroot', default='../result', help='path to saving.')
    FLAG.add_argument('--dataroot', default='data', help='path to dataset.')
    FLAG.add_argument('--manual_seed', default=42, help='manual seed.')
    FLAG.add_argument('--p1', default=1.0, type=float, help='p1, propotion of complementary label, 1.0 means full complementary labels')# p1
    FLAG.add_argument('--p2', default=1.0, type=float, help='p2, propotion of labeled data, 1.0 means use all labeled data')          #p2
    FLAG.add_argument('--image_size', default=32, help='image size.')
    FLAG.add_argument('--batch_size', default=128, help='batch size.')
    FLAG.add_argument('--num_workers', default=2, help='num workers.')
    FLAG.add_argument('--num_epoches', default=20, type=int, help='num workers.')
    FLAG.add_argument('--nc', default=1, type=int, help='channel of input image. MNIST:1 ; CIFAR10:3')
    FLAG.add_argument('--nz', default=64, help='length of noize.')
    FLAG.add_argument('--ndf', default=64, help='number of filters.')
    FLAG.add_argument('--ngf', default=64, help='number of filters.')
    arguments = FLAG.parse_args()
    return arguments


##############################################################################





assert torch.cuda.is_available(), '[!] CUDA required!'





def embed_z(opt):
    fixed = Variable(torch.Tensor(100, opt.nz).normal_(0, 1)).cuda()
    return fixed



def train_gan(opt):
    
    os.makedirs(os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/' + str(opt.p1) +  '_chkpts'), exist_ok=True)
    if not os.path.exists(os.path.join(opt.savingroot,opt.data_r,'data','processed/training'+str(opt.p1)+str(opt.p2)+'.pt')):
        generate_c_data(opt)
    else:
        print('data exits')

    #Build network
    if opt.data_r == 'MNIST':
        netd_c = LeNet5(opt.ndf, opt.nc, num_classes=10).cuda()
        netd_glj = D_MNIST(opt.ndf, opt.nc, num_classes=10).cuda()
    elif opt.data_r == 'CIFAR10':
        netd_c = ResNet18().cuda()#DPN92().cuda()#D_CIFAR10(opt.ndf, opt.nc, num_classes=10).cuda()



    optd_c = optim.SGD(netd_c.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)#optim.Adam(netd_c.parameters(), lr=0.0002, betas=(0.5, 0.999),weight_decay=5e-4)  #

    optd_glj = optim.SGD(netd_glj.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)#optim.Adam(netd_c.parameters(), lr=0.0002, betas=(0.5, 0.999),weight_decay=5e-4)  #

    print('training_start')
    step1 = 0
    step2 = 0
    acc1 = []
    acc2 = []

    # configure(os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/' +  str(opt.p1) + '/logs'), flush_secs=5)
    test_loader = torch.utils.data.DataLoader(     
        CIFAR10_Complementary(os.path.join(opt.savingroot,opt.data_r,'data'), train=False,unlabel=False, transform=transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])), batch_size=128, num_workers=2)
    
    fixed = embed_z(opt)
    dataset = CIFAR10_Complementary(os.path.join(opt.savingroot, opt.data_r, 'data'),train=True,unlabel=False, transform=tsfm, p1=opt.p1,p2=opt.p2)
    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2,worker_init_fn=np.random.seed) 
    loader1=loader
    loader2=loader
  #####################unlabel data   ##################################################
    unlabeldataset=CIFAR10_Complementary(os.path.join(opt.savingroot,opt.data_r,'data'), train=False,unlabel=True,transform=transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),p1=opt.p1,p2=opt.p2)
    unlabel_loader = torch.utils.data.DataLoader( unlabeldataset,batch_size=128, num_workers=2)
    unlabel_loader1=unlabel_loader
    unlabel_loader2=unlabel_loader
  #####################                ##################3###############################    
    choose_dataset10,choose_dataset11,choose_dataset20,choose_dataset21=torch.Tensor([]).cuda(),torch.Tensor([]).cuda(),torch.Tensor([]).cuda(),torch.Tensor([]).cuda()
    co_data1,co_data2=torch.Tensor([]).cuda(),torch.Tensor([]).cuda()
    for epoch in range(opt.num_epoches):
        print('Epoch:{:03d}'.format(epoch))

        if epoch % int(opt.num_epoches / 3) == 0 and epoch != 0:
            for param_group in optd_c.param_groups:
                param_group['lr'] = param_group['lr'] / 10
                print(param_group['lr'])
            for param_group in optd_glj.param_groups:
                param_group['lr'] = param_group['lr'] / 10
                print(param_group['lr'])


  #################model1
        step1 = train_c(epoch,netd_c, optd_c, loader1, step1,opt,co_data=co_data1) #
        accur1=test_acc(netd_c, test_loader , opt,False) #
        ''' 
        if epoch>5:
            choose_data2,new_unlabel_data1=test_acc(netd_c, unlabel_loader1 , opt,True)
            unlabeldataset=CIFAR10_Complementary(os.path.join(opt.savingroot,opt.data_r,'data'), train=False,unlabel=True,transform=transforms.Compose([
                transforms.Resize(opt.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),p1=opt.p1,p2=opt.p2,new_unlabeldata=new_unlabel_data1)
            new_unlabel_loader1 = torch.utils.data.DataLoader( unlabeldataset,batch_size=128, num_workers=2)
            unlabel_loader1=new_unlabel_loader1###############################
            print(choose_data2[0].size(),choose_data2[1].size(),new_unlabel_data1[0].size(),new_unlabel_data1[1].size())
        '''  
        acc1.append(accur1) 
  #################model2
        step2 = train_c(epoch,netd_glj, optd_glj, loader2, step2,opt,co_data=co_data2) #
        accur2=test_acc(netd_glj, test_loader , opt,False) 
        '''   
        if epoch>5:
            choose_data1,new_unlabel_data2=test_acc(netd_glj, unlabel_loader2 , opt,True)
            unlabeldataset=CIFAR10_Complementary(os.path.join(opt.savingroot,opt.data_r,'data'), train=False,unlabel=True,transform=transforms.Compose([
                transforms.Resize(opt.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),p1=opt.p1,p2=opt.p2,new_unlabeldata=new_unlabel_data2)
            new_unlabel_loader2 = torch.utils.data.DataLoader( unlabeldataset,batch_size=128, num_workers=2)
            unlabel_loader2=new_unlabel_loader2
            print(choose_data1[0].size(),choose_data1[1].size(),new_unlabel_data2[0].size(),new_unlabel_data2[1].size())
        '''    
        acc2.append(accur2)
        '''    
        if epoch>5:
   ####################co_data1
            choose_dataset10 = torch.cat([choose_dataset10,choose_data1[0]],dim=0)
            choose_dataset11 = torch.cat([choose_dataset11,choose_data1[1]],dim=0)
            print(choose_dataset10.size(),choose_dataset11.size())
            co_data1=[choose_dataset10,choose_dataset11]
            #new_traindata1=CIFAR10_Complementary(os.path.join(opt.savingroot, opt.data_r, 'data'),unlabel=False, transform=tsfm, p1=opt.p1,p2=opt.p2,co_data=co_data1)
   ####################co_data2
            choose_dataset20 = torch.cat([choose_dataset20,choose_data2[0]],dim=0)
            choose_dataset21 = torch.cat([choose_dataset21,choose_data2[1]],dim=0)
            print(choose_dataset20.size(),choose_dataset21.size())       
            co_data2=[choose_dataset20,choose_dataset21]
            #new_traindata2=CIFAR10_Complementary(os.path.join(opt.savingroot, opt.data_r, 'data'),unlabel=False, transform=tsfm, p1=opt.p1,p2=opt.p2,co_data=co_data2)
   ####################        
        #new_train_loader1= torch.utils.data.DataLoader(new_traindata2, batch_size=opt.batch_size, shuffle=True, num_workers=2,worker_init_fn=np.random.seed)
        #loader1=new_train_loader1
        #new_train_loader2= torch.utils.data.DataLoader(new_traindata1, batch_size=opt.batch_size, shuffle=True, num_workers=2,worker_init_fn=np.random.seed)
        #loader2=new_train_loader2
        '''    
    f = open(os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/'  + 'LeNet5_acc1.txt'), 'w')
    for cont in acc1:
        f.writelines(str(cont) + '\n')

    f.close()

    f = open(os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/'  + 'DMNIST_acc2.txt'), 'w')
    for cont in acc2:
        f.writelines(str(cont) + '\n')

    f.close()


if __name__ == '__main__':

    opt = args()
    opt.data_r = opt.dataset



    if opt.data_r == 'MNIST':                 ##   
        tsfm = transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:                                   ##   
        tsfm = transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    torch.cuda.manual_seed(opt.manual_seed)
    opt.dataset = os.path.join(opt.dataset, opt.dataset + '_' + str(opt.p2))
     #opt.datasetNIST/MNIST_1.0 
    configure(os.path.join(opt.savingroot, opt.dataset, str(opt.p1 * 100) + '%complementary/' +'/logs'),flush_secs=5)

##tensorboard_logger.configure(logdir, flush_secs=2)
##Configure logging: a file will be written to logdir, and flushed every flush_secs

    train_gan(opt)

