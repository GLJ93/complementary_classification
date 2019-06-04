
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
    probt = F.softmax(x,dim=1) #dim=1,��ÿһ�н���softmax��dim=0����ÿһ�н���softmax, eg,x:(torch.size([128,10])),dim=1���ά��1����ÿ��
    prob = torch.mm(probt, Q)
    out = torch.log(prob)
    loss = F.nll_loss(out, target) #nn.CrossEntropyLoss()����logsoftmax��nllloss������

    return loss

