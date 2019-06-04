
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from loss import forward_loss
from tensorboard_logger import configure, log_value
import torchvision
import os
#import matplotlib.pyplot as plt
import numpy as np
import time
import heapq#
import math

def denorm(x):
    return (x +1)/2

def train_c(epoch,netd_c,optd_c,loader,step,opt,co_data=torch.Tensor([]).cuda()):

    data_iter = iter(loader)
    iters = len(loader)
    extra_iters = 0
    if len(co_data)>0:
        if len(co_data[0])%128 == 0:
            extra_iters = math.floor(len(co_data[0])/128) 
        else:
            extra_iters = math.floor(len(co_data[0])/128) + 1
        co_label=co_data[1]
        co_datas=co_data[0]

    netd_c.train()
    i=0
    print('iters:',iters,'extra_iters:',extra_iters,'sum of both:',iters+extra_iters)
    while i < iters+extra_iters:
    #for _, (image_c, label) in enumerate(loader):

        # plt.imshow(np.transpose((image[0].cpu().numpy()+1)/2,[1,2,0]))
        # plt.show()
        # print(image.max(),image.min())  
        real_loss_c = torch.zeros(1).cuda()              
        if i<iters:
            labeled_data = data_iter.next()
            image_c,label = labeled_data
        
            index = label[:, 1]
            index = Variable(index).cuda()
            label = label[:, 0]

            real_label = label.cuda()

            if sum(index==1)+ sum(index == 0)>0:

                real_input_c = Variable(image_c).cuda()

                _, real_cls = netd_c(real_input_c)
             
                if sum(index == 1) > 0:
                    real_loss_c += F.cross_entropy(real_cls[index == 1], real_label[index == 1])  #
                if sum(index == 0) > 0:
                    real_loss_c += forward_loss(real_cls[index == 0], real_label[index == 0])  #

                optd_c.zero_grad()
                real_loss_c.backward()
                optd_c.step()
            i+=1
        else:
            if i != iters+extra_iters-1:
                co_labels = co_label[128*(i-iters):(127+128*(i-iters)), 0]
                real_input_c = Variable(co_datas[128*(i-iters):(127+128*(i-iters))]).cuda()
                print(co_labels.size(),real_input_c.size())
                _, real_cls = netd_c(real_input_c)
                real_loss_c += F.cross_entropy(real_cls, co_labels.long())
            else:
                co_labels = co_label[128*(i-iters):, 0]
                print(co_labels,co_datas[128*(i-iters):].size())
                real_input_c = Variable(co_datas[128*(i-iters):]).cuda()
                _, real_cls = netd_c(real_input_c)
                real_loss_c += F.cross_entropy(real_cls, co_labels.long())
            optd_c.zero_grad()
            real_loss_c.backward()
            optd_c.step()
            i+=1
            
    torch.save(netd_c.state_dict(), os.path.join(opt.savingroot, opt.dataset,
                                                 str(opt.p1 * 100) + '%complementary/' + str(
                                                     opt.p1) + '_chkpts/d_epoch{:03d}.pth'.format(epoch)))

    return step

def co_train_c(epoch,netd_c1, optd_c1,netd_c2, optd_c2,loader,unlabel_loader,step,opt):


    netd_c1.train()# 将本层及子层的training设定为True
    netd_c2.train()
    min_dataloader = min(len(loader), len(unlabel_loader))
    max_dataloader = max(len(loader), len(unlabel_loader))
    data_iter = iter(loader)
    unlabel_data_iter = iter(unlabel_loader)
    
    i=0
    #for _, (image_g,image_c, label) in enumerate(loader):

        # image_c:torch.size([128,1,32,32])
        # label:torch.size([128,2])
        
    while i < min_dataloader:
        real_loss_c =torch.zeros(1).cuda()
              
        labeled_data = data_iter.next()   # labeled data (include 0,1 labels)
        image_g,image_c, label = labeled_data
        index = label[:, 1]
        index = Variable(index).cuda()
        label = label[:, 0]
        real_label = label.cuda()
                                   
        if sum(index==1)+ sum(index == 0)>0:

            real_input_c = Variable(image_c).cuda()
            _, real_cls1 = netd_c1(real_input_c) #real_cls:torch.size([128,10])
            _, real_cls2 = netd_c2(real_input_c) 
            
            if sum(index == 1) > 0:
                real_loss_c += F.cross_entropy(real_cls1[index == 1], real_label[index == 1])  #  
                real_loss_c += F.cross_entropy(real_cls2[index == 1], real_label[index == 1]) 
                if i%20==0:
                    print(real_loss_c)         
            if sum(index == 0) > 0: ### I have 'elif' changed to 'if' here
                real_loss_c += forward_loss(real_cls1[index == 0], real_label[index == 0])  #
                real_loss_c += forward_loss(real_cls2[index == 0], real_label[index == 0])
                if i%20==0:
                    print(real_loss_c)
                       
        unlabel_data = unlabel_data_iter.next() # unlabeled data (-1 label)
        image_u,label_u = unlabel_data    
        unlabel_input_u = Variable(image_u).cuda()            
        _, unlabel_cls1 = netd_c1(unlabel_input_u) #unlabel_cls:torch.size([128,10])
        _, unlabel_cls2 = netd_c2(unlabel_input_u) 
                           
        probt1 = F.softmax(unlabel_cls1,dim=1)
        out1 = torch.log(probt1)
        probt2 = F.softmax(unlabel_cls2,dim=1)  
        out2 = torch.log(probt2)  
        Qx = (out1+out2)/2
        T=(probt1+probt2)/2
        KLDiv = sum(sum( (probt1.mul(out1-Qx) + probt2.mul(out2-Qx))/2 ))/(len(probt1))
        #KLDiv = sum(sum( probt1.mul(out1-out2) ))/(len(probt1))*0.5
            #KLDiv = sum(sum(-T*Qx-( -probt1*out1 - probt2*out2 )/2))/(len(probt1))
        if i%20==0:
            print(KLDiv)
        real_loss_c = real_loss_c+KLDiv
            #real_loss_c +=( F.kl_div((out1+out2)/2,probt1) + F.kl_div((out1+out2)/2,probt2 ) )/2
            #real_loss_c +=( torch.nn.KLDivLoss(out1,(probt1+probt2)/2) + torch.nn.KLDivLoss(out2,(probt1+probt2)/2) )/2
            
        optd_c1.zero_grad()
        optd_c2.zero_grad()
        real_loss_c.backward()
        optd_c1.step()
        optd_c2.step()
        i+=1
      

    torch.save(netd_c1.state_dict(), os.path.join(opt.savingroot, opt.dataset,
                                                 str(opt.p1 * 100) + '%complementary/' + str(
                                                     opt.p1) + '_chkpts/d1_epoch{:03d}.pth'.format(epoch)))
    torch.save(netd_c2.state_dict(), os.path.join(opt.savingroot, opt.dataset,
                                                 str(opt.p1 * 100) + '%complementary/' + str(
                                                     opt.p1) + '_chkpts/d2_epoch{:03d}.pth'.format(epoch)))
    return step

def test(netg,fixed,epoch,opt):
    netg.eval()

    fixed = Variable(torch.Tensor(100, opt.nz).normal_(0, 1)).cuda()
    label = Variable(torch.LongTensor([range(10)] * 10)).view(-1).cuda()

    fixed_input = netg(fixed,label)

    torchvision.utils.save_image(denorm(fixed_input.data), os.path.join(opt.savingroot,opt.dataset,str(opt.p1 * 100) + '%complementary/' + str(opt.p1)+'_images/fixed_{epoch:03d}.jpg'), nrow=10)
#new_testdata = c_unlabel_data = torch.Tensor([]).cuda()
def test_acc(model, test_loader,opt,unlabel=True):

    model.eval()
    test_loss = 0
    correct = 0
    
    if unlabel:

        choose_unlabel_data=torch.Tensor([]).cuda()
        choose_target = torch.Tensor([]).cuda()
        plabels = torch.Tensor([]).cuda()
        newdata = torch.Tensor([]).cuda()
        newtarget = torch.Tensor([]).cuda()
        max_prob = []
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            
            output = model(data)[1]   
            pred = output.max(1, keepdim=True)[1] # get Pseudo labels torch.size([128,1]) 
            set_pred = set(pred.view(-1).tolist())                            
            prob=(output.max(1, keepdim=True)[0]).view(-1) #torch.size([128])            
            #prob2=prob.detach().cpu().numpy()
            
            for i in list(set_pred):
                prob_i=prob[pred.view(-1)==i].detach().cpu().numpy()
                data_i=data[pred.view(-1)==i]
                target_i=target[pred.view(-1)==i]
                max_prob.extend(heapq.nlargest(1, prob_i))#the choosed max prob
                maxpro_index = heapq.nlargest(1, range(len(prob_i)), prob_i.take)#the index of max prob                               
                choose_unlabel_data = torch.cat([choose_unlabel_data,data_i[maxpro_index]],dim=0)
                choose_target = torch.cat([choose_target,target_i[maxpro_index].float()],dim=0)
                
                plabels = torch.cat([plabels,torch.Tensor([[i,1]]).cuda()],dim=0)
                
                data_index = set([j for j in range(data_i.size()[0])])
                maxpro_index = set(maxpro_index)
                Cindex = list(data_index-maxpro_index)
                newdata = torch.cat([newdata,data_i[Cindex]],dim=0)  
                newtarget = torch.cat([newtarget,target_i[Cindex].float()],dim=0)
            #maxpro_index = heapq.nlargest(2, range(len(prob2)), prob2.take)
            #choose_unlabel_data = torch.cat([choose_unlabel_data,data[maxpro_index]],dim=0)
            #choose_target = torch.cat([choose_target,target[maxpro_index].float()],dim=0)
            
            #plabel = pred[maxpro_index]   #           
            #pindex=torch.ones(plabel.size()[0]).long().cuda()
            #pindex=pindex.unsqueeze(1)
            #plabelss = torch.cat([plabel, pindex], dim=1).float()
            #plabels = torch.cat([plabels,plabelss],dim=0)                      
           
            #data_index = set([i for i in range(data.size()[0])])
            #maxpro_index = set(maxpro_index)
            #Cindex = list(data_index-maxpro_index)
            #newdata = torch.cat([newdata,data[Cindex]],dim=0)            
            #newtarget = torch.cat([newtarget,target[Cindex].float()],dim=0)
        c_unlabel_data = [choose_unlabel_data,plabels]
        new_testdata = [newdata,newtarget]
        
        choose_unlabel_data_10 = torch.Tensor([]).cuda()
        plabels_10=torch.Tensor([]).cuda()            
        max_prob = torch.Tensor(max_prob)
        set_plabels = set(plabels[:,0].view(-1).tolist())
        for k in list(set_plabels):
            max_prob_k = max_prob[plabels[:,0].view(-1)==k].numpy()
            choose_unlabel_data_k = choose_unlabel_data[plabels[:,0].view(-1)==k]
            choose_target_k = choose_target[plabels[:,0].view(-1)==k]
            maxpro_index_k = heapq.nlargest(1, range(len(max_prob_k)), max_prob_k.take)
            choose_unlabel_data_10 = torch.cat([choose_unlabel_data_10,choose_unlabel_data_k[maxpro_index_k]],dim=0)
            plabels_10 = torch.cat([plabels_10,torch.Tensor([[k,1]]).cuda()],dim=0)
            
            data_index = set([j for j in range(choose_unlabel_data_k.size()[0])])
            maxpro_index_k = set(maxpro_index_k)
            Cindex = list(data_index-maxpro_index_k)
            newdata_left = choose_unlabel_data_k[Cindex]
            newtarget_left = choose_target_k[Cindex]
            newtarget = torch.cat([newtarget,newtarget_left],dim=0)
            newdata = torch.cat([newdata,newdata_left],dim=0)
        #maxpro_index_20 = heapq.nlargest(20, range(len(max_prob)), max_prob.take)
        
        #choose_unlabel_data_20 = choose_unlabel_data[maxpro_index_20]
        #plabels_20 = plabels[maxpro_index_20]
        
        #data_index = set([i for i in range(choose_unlabel_data.size()[0])])
        #maxpro_index_20 = set(maxpro_index_20)
        #Cindex = list(data_index-maxpro_index_20)
        #newdata_left = choose_unlabel_data[Cindex]
        #newtarget_left = choose_target[Cindex]
        #newtarget = torch.cat([newtarget,newtarget_left],dim=0)
        #newdata = torch.cat([newdata,newdata_left],dim=0)
        c_unlabel_data_10 = [choose_unlabel_data_10,plabels_10]
        new_testdata = [newdata,newtarget]
               
    else:
        for data, target in test_loader:
            
            data, target = data.cuda(), target.cuda()
            output = model(data)[1]
            pred = output.max(1, keepdim=True)[1]    
            test_loss += F.cross_entropy(output, target).sum().item() # sum up batch loss
            correct += pred.eq(target.view_as(pred)).sum().item()
#    print(c_unlabel_data.size(),new_testdata.size())
#    input()        
    if unlabel:      
        return c_unlabel_data_10,new_testdata        
    else:
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)*1.0))
        return correct / len(test_loader.dataset)*1.0
