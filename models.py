import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F


class G_MNIST(nn.Module):

    def __init__(self, nz, ngf, nc):
        super(G_MNIST, self).__init__()

        self.embed = nn.Embedding(10, nz)

        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(ngf * 8)

        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ngf * 4)

        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 1, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ngf * 1)


        self.conv5 = nn.ConvTranspose2d(ngf * 1, nc, 4, 2, 1)

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

        self.__initialize_weights()

    def forward(self, z,label):
        input = z.mul_(self.embed(label))
        x = input.view(input.size(0), -1, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)


        x = self.conv5(x)
        output = self.tanh(x)
        return output

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


class D_MNIST(nn.Module):

    def __init__(self, ndf, nc, num_classes=10):
        super(D_MNIST, self).__init__()
        self.ndf = ndf
        self.lrelu = nn.ReLU()
#'''
#        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1)

#        self.conv3 = nn.Conv2d(ndf , ndf * 4, 4, 2, 1)
#        self.bn3 = nn.BatchNorm2d(ndf * 4)
#        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
#        self.bn4 = nn.BatchNorm2d(ndf * 8)
#        self.conv5 = nn.Conv2d(ndf * 8, ndf * 1, 4, 1, 0)
#        self.gan_linear = nn.Linear(ndf * 1, 1)
#        self.aux_linear = nn.Linear(ndf * 1, num_classes)
#'''
        self.conv1 = nn.Conv2d(nc, ndf, 5, 1, 0)

        self.conv3 = nn.Conv2d(ndf , ndf * 2, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf * 2)
        self.conv4 = nn.Conv2d(ndf * 2, ndf * 4, 5, 1, 0)
        self.bn4 = nn.BatchNorm2d(ndf * 4)
        self.conv5 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf * 8)

        self.conv6 = nn.Conv2d(ndf * 8, ndf * 1, 5, 1, 0)
        self.gan_linear = nn.Linear(ndf * 1, 1)
        self.aux_linear = nn.Linear(ndf * 1, num_classes)

        self.sigmoid = nn.Sigmoid()
        self.__initialize_weights()

    def forward(self, input):

        x = self.conv1(input)
        x = self.lrelu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.lrelu(x)

        x = self.conv6(x)
        x = x.view(-1, self.ndf * 1)
        c = self.aux_linear(x)

        s = self.gan_linear(x)
        s = self.sigmoid(s)
        return s.squeeze(1), c.squeeze(1)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
#######################################################################
class glj_MNIST(nn.Module):   

    def __init__(self, ndf, nc, num_classes=10): 
        super(glj_MNIST, self).__init__()
        self.ndf = ndf
        self.lrelu = nn.ReLU()
        self.conv1 = nn.Conv2d(nc, ndf, 3, 1, 0)

        self.conv3 = nn.Conv2d(ndf , ndf * 2, 5, 1, 0) 
        self.bn3 = nn.BatchNorm2d(ndf * 2)

        self.conv4 = nn.Conv2d(ndf * 2, ndf * 4, 7, 1, 0)
        self.bn4 = nn.BatchNorm2d(ndf * 4)

        self.conv5 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf * 8)

        self.conv6 = nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1)
        self.bn6 = nn.BatchNorm2d(ndf * 8)

        self.conv7 = nn.Conv2d(ndf * 8, ndf*1, 5, 1, 0)
        self.gan_linear = nn.Linear(ndf , 1)
        self.aux_linear = nn.Linear(ndf, num_classes)

        self.sigmoid = nn.Sigmoid()
        self.__initialize_weights()

    def forward(self, input):

        x = self.conv1(input)  #30*30*64
        x = self.lrelu(x)
       
        x = self.conv3(x)     #26*26*128
        x = self.bn3(x)
        x = self.lrelu(x)
       
        x = self.conv4(x)
        x = self.bn4(x)      #20*20*256
        x = self.lrelu(x)
       
        x = self.conv5(x)
        x = self.bn5(x)      #10*10*512
        x = self.lrelu(x)

        x = self.conv6(x)
        x = self.bn6(x)      #5*5*256
        x = self.lrelu(x)
        
        x = self.conv7(x)    #1*1*64
        x = x.view(-1, self.ndf)  #1*64
        c = self.aux_linear(x)       #1*10
        
        
        s = self.gan_linear(x)     #1*1
        s = self.sigmoid(s)
        return s.squeeze(1), c.squeeze(1)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
##########################################################################

class LeNet5(nn.Module):    

    def __init__(self, ndf, nc, num_classes=10): 
        super(LeNet5, self).__init__()
        self.ndf = ndf
        self.lrelu = nn.ReLU()
        self.conv1 = nn.Conv2d(nc, 6, 5, 1, 0)

        self.conv3 = nn.Conv2d(6 , 6, 4, 2, 1) 
        self.bn3 = nn.BatchNorm2d(6)
        self.conv4 = nn.Conv2d(6, 16, 5, 1, 0)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 120, 5, 1, 0)
        self.linear7 = nn.Linear(120, 84)
        self.linear8 = nn.Linear(84, num_classes)

        self.sigmoid = nn.Sigmoid()
        self.__initialize_weights()

    def forward(self, input):

        x = self.conv1(input)  #28*28*6
        x = self.lrelu(x)
 
        x = self.conv3(x)     #14*14*6
        x = self.bn3(x)
        x = self.lrelu(x)

        x = self.conv4(x)
        x = self.bn4(x)      #10*10*6
        x = self.lrelu(x)

        x = self.conv5(x)
        x = self.bn5(x)      #5*5*16
        x = self.lrelu(x)

        x = self.conv6(x)    #1*1*120
        x = x.view(-1, 120 )  #1*120
        c = self.linear7(x)       #1*84
        c = self.linear8(c)  #1*10

        
        s = self.sigmoid(x)
        return s.squeeze(1), c.squeeze(1)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

class G_CIFAR10(nn.Module):

    def __init__(self, nz, ngf, nc):
        super(G_CIFAR10, self).__init__()
        self.embed = nn.Embedding(10, nz)
        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(ngf * 8)

        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ngf * 4)

        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ngf * 2)

        self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ngf * 1)

        self.conv5 = nn.ConvTranspose2d(ngf, ngf * 1, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(ngf * 1)

        self.conv6 = nn.Conv2d(ngf * 1, nc, 3, 1, 1)

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

        self.__initialize_weights()

    def forward(self, z,label):
        input = z.mul_(self.embed(label))
        x = input.view(input.size(0), -1, 1, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.conv6(x)
        output = self.tanh(x)
        return output

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


class D_CIFAR10(nn.Module):

    def __init__(self, ndf, nc, num_classes=10):
        super(D_CIFAR10, self).__init__()
        self.ndf = ndf
        self.lrelu = nn.ReLU()
        self.conv0 = nn.Conv2d(nc, ndf, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(ndf)
        self.conv1 = nn.Conv2d(ndf, ndf, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 1, 4, 1, 0)
        self.gan_linear = nn.Linear(ndf * 1, 1)
        self.aux_linear = nn.Linear(ndf * 1, num_classes)

        self.sigmoid = nn.Sigmoid()
        self.__initialize_weights()

    def forward(self, input):

        x = self.conv0(input)
        x = self.bn0(x)
        x = self.lrelu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu(x)

        x = self.conv5(x)
        x = x.view(-1, self.ndf * 1)
        c = self.aux_linear(x)

        s = self.gan_linear(x)
        s = self.sigmoid(s)
        return s.squeeze(1), c.squeeze(1)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


'''Dual Path Networks in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes+dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, out_planes+dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes+dense_depth)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        x = self.shortcut(x)
        d = self.out_planes
        out = torch.cat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)
        out = F.relu(out)
        return out


class DPN(nn.Module):
    def __init__(self, cfg):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.last_planes = 64
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.aux_linear = nn.Linear(out_planes[3]+(num_blocks[3]+1)*dense_depth[3], 10)
        self.gan_linear = nn.Linear(out_planes[3]+(num_blocks[3]+1)*dense_depth[3], 1)

        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i==0))
            self.last_planes = out_planes + (i+2) * dense_depth
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        c = self.aux_linear(out)

        s = self.gan_linear(out)
        s = self.sigmoid(s)
        return s.squeeze(1), c.squeeze(1)


def DPN26():
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (2,2,2,2),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg)

def DPN92():
    cfg = {
        'in_planes': (96,192,384,768),
        'out_planes': (256,512,1024,2048),
        'num_blocks': (3,4,20,3),
        'dense_depth': (16,32,24,128)
    }
    return DPN(cfg)


def test():
    net = DPN92()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.aux_linear = nn.Linear(512*block.expansion, num_classes)
        self.gan_linear = nn.Linear(512*block.expansion, 1)

        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        c = self.aux_linear(out)
        s = self.gan_linear(out)
        s = self.sigmoid(s)
        return s.squeeze(1), c.squeeze(1)


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
