import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation,selfAttention


class get_model(nn.Module):
    def __init__(self, num_classes,dropout_rate):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)  ##6##
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        

    def forward(self, xyz):
        l0_points = xyz#torch.Size([4, 9, 2048]) 
        l0_xyz = xyz[:,:3,:]#torch.Size([4, 9, 2048]) torch.Size([4, 3, 2048])
        # print(l0_points.shape,l0_xyz.shape)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)#Size([4, 64, 1024]) ([4, 3, 1024])
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)#torch.Size([4, 4, 2048])
        # print(x.shape)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)#torch.Size([4, 2048, 4])
        # print(x.shape)
        return x, l4_points
    

def init_weights(module):
    class_name = module.__class__.__name__
    if class_name.find("Conv2") != -1 or class_name.find("Linear") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif class_name.find("Norm2d") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)

class ConvBlock(nn.Module):
    """Convolution Block"""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 padding_mode="zeros", bias=False, norm="batch", activation="relu"):
        super().__init__()

        if norm == 'batch':
            norm_layer = nn.BatchNorm1d#对batchsize张图片对应通道对应像素进行均值方差，对每个像素进行跟新
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.Identity

        if activation == "relu":
            activation_layer = nn.ReLU(inplace=True)
        elif activation == "leaky":
            activation_layer = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "hardswish":
            activation_layer = nn.Hardswish(inplace=True)
        else:
            activation_layer = nn.Identity()

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, bias=bias),#, stride=stride, padding=padding,padding_mode=padding_mode, bias=bias

            norm_layer(out_ch),
            activation_layer
        )
    
    def forward(self, x):
        # print('&&&')
        return self.net(x)   
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
# from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
from scipy import signal
class Discriminator(nn.Module):
    """Down Sampling Discriminator"""
    def __init__(self, n_f, in_ch=64):#, device=torch.device("cpu")
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(in_ch*2, n_f, 4, stride=2, activation="leaky"),    
            ConvBlock(n_f, n_f*2, 1, stride=2, activation="leaky"),   
            ConvBlock(n_f*2, n_f*4, 1, stride=2,  activation="leaky"),  
            ConvBlock(n_f*4, n_f*8, 1, stride=2,activation="leaky"),  
            nn.Conv1d(n_f*8, 1, 1)
        )
        # self.net.to(device)
        self.net.cuda()
        self.net.apply(init_weights)
    def forward(self, x):
        x = x.contiguous().view(64, 128,4)
        return self.net(x)
class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()
        self.register_buffer("real_label", torch.ones(1))
        self.register_buffer("fake_label", torch.zeros(1))

    def __call__(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label.expand_as(prediction)
        else:
            target_tensor = self.fake_label.expand_as(prediction)
        loss = self.loss(prediction, target_tensor)
        return loss

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):        
        weights  = torch.ones([pred.shape[0],pred.shape[1]])#torch.float32
        a = 5
        weights[:,3][torch.where(target == 0)] = a
        weights[:,3][torch.where(target == 1)] = 4*a
        # weights[:,3][torch.where(target == 2)] = 2
        # weights[:,3][torch.where(target == 3)] = 2
        # print(weight.max())
        nx = 3
        # kernel = torch.ones(nx)/nx
        ny = 1
        kernel = torch.ones((nx,ny))*1/(nx*ny)
        # print(kernel,kernel.shape)
        # print(weights[20:1000,3])
        weights = signal.convolve(weights,kernel,mode='same')
        weights = torch.from_numpy(weights).cuda()
        # print(weights[20:1000,3])
        # print(pred_new.shape,target.shape)#torch.Size([32768, 4]) torch.Size([32768])
        loss = F.nll_loss(pred, target, weight = weight, reduction="none")#, reduction="none" (N,1) 
        # print(target)# (N, ...)  
        # print(weights[:,3])   
        loss = loss*weights[:,3]
        # print(loss)
        loss = torch.mean(loss)
        # print(loss)
        return loss
# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()
#     def forward(self, pred, target, trans_feat, weight):
#        # print(pred.shape,target.shape)
#         total_loss = F.nll_loss(pred, target, weight=weight)
#         return total_loss

# if __name__ == '__main__':
#     import  torch
#     model = get_model(13)
#     xyz = torch.rand(6, 9, 2048)
#     (model(xyz))