
import torch.nn.functional as F
import torch
import numpy as np

def complementary_transition_matrix():
    ncls = 10
    rho = 1.0
    M = (rho / (ncls - 1)) * np.ones((ncls, ncls))  #
    for i in range(ncls):
        M[i, i] = 1. - rho
    M = torch.from_numpy(M).float().cuda()
    return M

Q = complementary_transition_matrix()

def forward_loss(x, target):
    probt = F.softmax(x,dim=1) #dim=1,对每一行进行softmax；dim=0，对每一列进行softmax, eg,x:(torch.size([128,10])),dim=1针对维数1，即每行
    prob = torch.mm(probt, Q)
    out = torch.log(prob)
    loss = F.nll_loss(out, target) #nn.CrossEntropyLoss()包含logsoftmax，nllloss不包含

    return loss

