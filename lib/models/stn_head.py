from __future__ import absolute_import

import math
import numpy as np
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


def conv3x3_block(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

  block = nn.Sequential(
    conv_layer,
    nn.BatchNorm2d(out_planes),
    nn.ReLU(inplace=True),
  )
  return block


class STNHead(nn.Module):
  def __init__(self, in_planes, num_ctrlpoints, activation='none'):
    super(STNHead, self).__init__()

    self.in_planes = in_planes
    self.num_ctrlpoints = num_ctrlpoints
    self.activation = activation
    self.stn_convnet = nn.Sequential(
      conv3x3_block(in_planes, 32),  # 32*64
      nn.MaxPool2d(kernel_size=2, stride=2),
      conv3x3_block(32, 64),  # 16*32
      nn.MaxPool2d(kernel_size=2, stride=2),
      conv3x3_block(64, 128),  # 8*16
      nn.MaxPool2d(kernel_size=2, stride=2),
      conv3x3_block(128, 256),  # 4*8
      nn.MaxPool2d(kernel_size=2, stride=2),
      conv3x3_block(256, 256),  # 2*4,
      nn.MaxPool2d(kernel_size=2, stride=2),
      conv3x3_block(256, 256)  # shape=[batch_size, 256, 1, 2]
    )

    # [batch_size, 512] -> [batch_size, num_ctrlpoints*2]  e.g. num_ctrlpoints=20
    self.stn_fc1 = nn.Sequential(
                      nn.Linear(2*256, 512),
                      nn.BatchNorm1d(512),
                      nn.ReLU(inplace=True))
    self.stn_fc2 = nn.Linear(512, num_ctrlpoints*2)  # 40，应该是上20 下20

    self.init_weights(self.stn_convnet)
    self.init_weights(self.stn_fc1)
    self.init_stn(self.stn_fc2)

  def init_weights(self, module):
    for m in module.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()

  def init_stn(self, stn_fc2):
    margin = 0.01
    sampling_num_per_side = int(self.num_ctrlpoints / 2)  # 每边各10个点
    ctrl_pts_x = np.linspace(margin, 1.0 - margin, sampling_num_per_side)  # len(ctrl_pts_x)=10，所以说是算上了边界点的
    ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin  # [0.01, 0.01, ...] 共10个
    ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1-margin)  # [0.99, 0.99, ...] 共10个
    ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)  # shape=[10, 2]  每行代表了上边一排控制点的 x,y 坐标
    ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)  # shape=[10, 2]  每行代表了下边一排控制点的x, y坐标
    ctrl_points = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float32)  # shape=[20, 2] 前面10个是上边一排控制点的x,y坐标，后面10个是下边一排控制点的x,y坐标
    if self.activation is 'none':
      pass
    elif self.activation == 'sigmoid':
      ctrl_points = -np.log(1. / ctrl_points - 1.)
    stn_fc2.weight.data.zero_()
    stn_fc2.bias.data = torch.Tensor(ctrl_points).view(-1)

  def forward(self, x):
    x = self.stn_convnet(x)  # [batch_size, 256, 1, 2]
    batch_size, _, h, w = x.size()
    x = x.view(batch_size, -1)  # [batch_size, 512]
    img_feat = self.stn_fc1(x)  # [batch_size, num_ctrlpoints*2]
    # TODO 为什么要乘以0.1呢？
    x = self.stn_fc2(0.1 * img_feat)
    if self.activation == 'sigmoid':
      x = F.sigmoid(x)  # x的值归一化(0,1)之间
    x = x.view(-1, self.num_ctrlpoints, 2)  # [batch_size, num_ctrlpoints, 2]  求的是所有ctrl points的x, y坐标
    return img_feat, x


if __name__ == "__main__":
  in_planes = 3
  num_ctrlpoints = 20
  activation='none' # 'sigmoid'
  stn_head = STNHead(in_planes, num_ctrlpoints, activation)
  input = torch.randn(10, 3, 32, 64)
  control_points = stn_head(input)    
  print(control_points[1].size())  # [10, 20, 2]