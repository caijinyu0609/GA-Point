import numpy as np


def noise_Gaussian(points, mean):
    noise = np.random.normal(0, mean, points.shape)
    out = points + noise
    return out


def main(filepath,savepath,savepath2,savepath3):
    # 加载点云
    points = np.loadtxt(filepath)[:, 0:7]
    # 获取添加噪声的点云
    out = noise_Gaussian(points, 0.05)
    out2 = noise_Gaussian(points, 0.02)
    out3 = noise_Gaussian(points, 0.01)
    # 保存点云
    np.savetxt(savepath, out, fmt='%.6f', delimiter=' ')
    np.savetxt(savepath2, out2, fmt='%.6f', delimiter=' ')
    np.savetxt(savepath3, out3, fmt='%.6f', delimiter=' ')

import os
if __name__ == '__main__':
    path = "/data/caijinyu/ANAN/Cloud/Pointnet_Pointnet2_pytorch-master/data/drill"
    save = "/data/caijinyu/ANAN/Cloud/Pointnet_Pointnet2_pytorch-master/data/drill_noise"
    file = os.listdir(path)
    for name in file:
        filepath = os.path.join(path,name)
        savepath = os.path.join(save,name.replace('.txt','_0.05.txt'))
        savepath2 = os.path.join(save,name.replace('.txt','_0.02.txt'))
        savepath3 = os.path.join(save,name.replace('.txt','_0.01.txt'))
        main(filepath,savepath,savepath2,savepath3)

