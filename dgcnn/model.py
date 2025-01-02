#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

# 与DGCNN中的代码一致，得到k近邻下的xj-xi及相同shape扩充出来的xi
def get_graph_feature_base(x, k, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    return feature, x

# DGCNN原本的特征，xj-xi和xi拼接
def get_graph_feature(x, k=20, idx=None):
    feature, x = get_graph_feature_base(x, k, idx)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

def get_ti_hint(x: Tensor, eigen_k, eigval_only):
    """
    x: (batch_size, num_points, k, num_dims), meaning x_j-x_i in KNN
    eigen_k：选取前k大的特征值对应的结果
    eigenval_only：Bool, 是否只求特征值，不求特征向量。

    得到localized Gram矩阵的特征值和特征向量，过程不进行梯度传播。
    由于特征向量可能存在正负相，统一调整到第一个分量的signbit为0
    """
    batch_size, num_points, k, _ = x.shape # k is point num in a subgroup
    collapsed_size = batch_size * num_points
    with torch.no_grad():
        x = x.reshape((batch_size * num_points, *x.shape[2:]))
        y = x @ x.transpose(-2, -1)
        dim0 = torch.arange(collapsed_size).unsqueeze(1)
        if eigval_only:
            eigvals = torch.linalg.eigvalsh(y).to(torch.float)
            idx = torch.topk(torch.abs(eigvals), eigen_k)[1]
            eigvals = eigvals[dim0, idx]
            return eigvals.reshape((batch_size, num_points, eigen_k))

        eigvals, eigvecs = torch.linalg.eigh(y)
        eigvals = eigvals.to(torch.float)
        idx = torch.topk(torch.abs(eigvals), eigen_k)[1]
        eigvals = eigvals[dim0, idx]

        eigvecs = torch.gather(eigvecs.to(torch.float), 2,
                               idx.unsqueeze(-2).expand(-1, eigvecs.size(1), -1))
        inv_idx = torch.signbit(eigvecs[:, 0, :]).unsqueeze(1).expand_as(eigvecs)
        eigvecs[inv_idx] = -eigvecs[inv_idx]
        eigvecs = eigvecs.transpose(1, 2)

    return eigvals.reshape((batch_size, num_points, eigen_k)), \
           eigvecs.reshape((batch_size, num_points, eigen_k, k))

def get_naive_hint(x: Tensor, topk = None):
    """
    x: (batch_size, num_points, k, num_dims), meaning x_j-x_i in KNN
    topk: 选择前k大的值

    对于localized gram矩阵的下三角，选取最大的topk个值。
    """
    batch_size, num_points, k, _ = x.shape # k is point num in a subgroup
    with torch.no_grad():
        x = x.reshape((batch_size * num_points, *x.shape[2:]))
        y = x @ x.transpose(-2, -1)
        r, c = torch.tril_indices(k, k, offset=-1)
        feature = y[:, r, c]
    if topk is None:
        return feature.reshape((batch_size, num_points, -1))

    return feature.topk(topk)[0].reshape((batch_size, num_points, topk))

def get_graph_feature_with_ti_hint(x, k, eigen_k, idx=None, eigval_only=False, ti_hint_only=False):
    """
    get_graph_feature和get_ti_hint的组合。
    """
    feature, x = get_graph_feature_base(x, k, idx)
    diff = feature - x
    if eigval_only:
        return get_ti_hint(diff, eigen_k, eigval_only).transpose(1, 2)

    eigvals, eigvecs = get_ti_hint(diff, eigen_k, eigval_only)
    eigvals, eigvecs = eigvals.transpose(1, 2), eigvecs.transpose(1, 2)
    if ti_hint_only:
        return eigvals, eigvecs

    feature = torch.cat((diff, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, eigvals, eigvecs

def get_graph_feature_naive_hint(x, k, topk, idx=None):
    feature, x = get_graph_feature_base(x, k, idx)
    diff = feature - x
    return get_naive_hint(diff, topk).transpose(1, 2)

def get_brute_feature_with_centroid(x: Tensor):
    """
    x: (batch_size, 3, num_points), the initial pointcloud.
    直接使用点云中心的localized Gram Matrix。
    """
    y = x.mean(dim = 1, keepdim=True)
    diff = y - x # shape same as x
    feature = diff.transpose(1, 2) @ diff # (batch_size, num_points, num_points)
    return feature

class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class DGCNN_Centered(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_Centered, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        # 优化后的特征向量压缩
        with torch.no_grad():
            x = x - x.mean() # (batch_size, 3, N)
            feature = x @ x.transpose(1, 2) # (batch_size, 3, 3)
            eigvals, eigvecs = torch.linalg.eigh(feature) # (batch_size, 3, 3)
            x = eigvecs.transpose(1, 2) @ x

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

class DGCNN_Local(nn.Module):
    # 计算每个点的LGM
    @staticmethod
    def compute_point_differences(point_cloud):
        """
        Args:
            point_cloud: Tensor of shape (batch_size, 3, num_points)

        Returns:
            Tensor of shape (batch_size, num_points, 3, num_points)
        """
        batch_size, _, num_points = point_cloud.shape

        # Reshape the point cloud to (batch_size, 3, num_points, 1) for broadcasting
        point_cloud_expanded = point_cloud.unsqueeze(-1)  # Shape: (batch_size, 3, num_points, 1)

        # Reshape the point cloud to (batch_size, 3, 1, num_points) for broadcasting
        point_cloud_transposed = point_cloud.unsqueeze(2)  # Shape: (batch_size, 3, 1, num_points)

        # Subtract to get the differences
        differences = point_cloud_transposed - point_cloud_expanded  # Shape: (batch_size, 3, num_points, num_points)

        # Transpose the result to match the desired output shape (batch_size, num_points, 3, num_points)
        differences = differences.permute(0, 2, 1, 3)

        return differences

    def __init__(self, args, output_channels=40):
        super(DGCNN_Local, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x: Tensor):
        batch_size, _, num_points = x.shape
        with torch.no_grad():
            y: Tensor = self.compute_point_differences(x)
            feature = y @ y.transpose(2, 3) # (batch_size, num_points, 3, 3)
            eigvals, eigvecs = torch.linalg.eigh(feature) # (batch_size, num_points, 3, 3)
            y = eigvecs.transpose(2, 3) @ y  # (batch_size, num_points, 3, num_points)
            y_subtract = y[torch.arange(batch_size)[:, None], torch.arange(num_points), :,
                         torch.arange(num_points)]  # (batch_size, num_points, 3)
            y_subtract = y - y_subtract.unsqueeze(-1)

            # 得到旋转不变的坐标
            y = y.reshape((batch_size * num_points, 3, num_points))
            # KNN
            y_subtract = y_subtract.reshape((batch_size * num_points, 3, num_points))
            reduced_result = (y_subtract * y_subtract).sum(dim = 1)
            _, topk_indices = torch.topk(reduced_result, k=self.k, dim=1)

            # (batch_size * num_points, 3, self_k)
            topk_indices_expanded = topk_indices.unsqueeze(1).expand(-1, 3, -1)

            # (batch_size * num_points, 6, self_k)，对应于(xi, xj-xi)
            result = torch.concat(
                (torch.gather(y, dim=2, index=topk_indices_expanded),
                 torch.gather(y_subtract, dim=2, index=topk_indices_expanded)),1)

            x0 = result.reshape((batch_size, num_points, 6, self.k))
            x = x0.transpose(1, 2)

        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

# 以特征值和坐标一同作为第一层输入，后续不变
class DGCNN_Eigval(nn.Module):
    def __init__(self, args, output_channels=40):
        super().__init__()
        self.args = args
        self.k = args.k
        self.eigen_topk = args.eig_topk
        self.eigen_k = args.eig_knn_k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d((3 + self.eigen_topk) * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        ti_hint = get_graph_feature_with_ti_hint(x, self.eigen_k, self.eigen_topk,
                                                 eigval_only=True, ti_hint_only=True)
        x = torch.concat((x, ti_hint), dim=1)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

# 将每个点的localized Gram矩阵的topk特征向量拼接在第一层卷积后的特征上。
# 我们也尝试过用特征向量代替第一层的特征变换中的xj-xi，效果不明显。
class DGCNN_Eigvec(nn.Module):
    def __init__(self, args, output_channels=40):
        super().__init__()
        self.args = args
        self.k = args.k
        self.eigen_topk = args.eig_topk
        self.eigen_k = args.eig_knn_k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(3*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d((64+self.eigen_topk)*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512+self.eigen_topk, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        eigvals, eigvecs = get_graph_feature_with_ti_hint(
            x, self.eigen_k, self.eigen_topk, ti_hint_only=True)

        x = get_graph_feature(x, k=self.k)
        # x = torch.concat((eigvecs, x[:, 3:, :]), dim=1)
        x = self.conv1(x)
        # x1 = x.max(dim=-1, keepdim=False)[0]
        x1 = torch.concat((x.max(dim=-1, keepdim=False)[0], eigvecs.max(dim=-1, keepdim=False)[0]), dim=1)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

# 对点云减去中心坐标后，使用SVD的基变为旋转不变坐标；当eigen_topk != 0时会把naive的
# 特征拼接到第一层的输出中。
class DGCNN_PCA(nn.Module):
    def __init__(self, args, output_channels=40):
        super().__init__()
        self.args = args
        self.k = args.k
        self.eigen_topk = args.eig_topk
        self.eigen_k = args.eig_knn_k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d((64+ self.eigen_topk)*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512 + self.eigen_topk, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        if self.eigen_topk != 0:
            local_feature = get_graph_feature_naive_hint(x, self.eigen_k, self.eigen_topk)

        # PCA。
        if not self.args.only_naive:
            with torch.no_grad():
                x = x - x.mean() # centralized
                x = x.transpose(1, 2)
                U, S, VT = torch.pca_lowrank(x)
                # VT: batch_size * 3 * 3
                x = U @ torch.diag_embed(S)
                x = x.transpose(1, 2)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)

        if self.eigen_topk != 0:
            x1 = torch.concat((x.max(dim=-1, keepdim=False)[0], local_feature), dim=1)
        else:
            x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

# 使用点云中心的localized Gram Matrix，经过MLP缩小维度，再作为输入进入DGCNN。
class DGCNN_Brute(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_Brute, self).__init__()
        self.args = args
        self.k = args.k

        self.linear0 = nn.Sequential(
            nn.Linear(args.num_points, args.compact_feature, bias=False),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.conv1 = nn.Sequential(nn.Conv2d(args.compact_feature * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        with torch.no_grad():
            x = get_brute_feature_with_centroid(x)

        x = self.linear0(x)
        x = x.transpose(1, 2)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
