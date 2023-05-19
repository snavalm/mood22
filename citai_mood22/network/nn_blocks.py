
import math

from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvDropoutNormNonlin(nn.Module):
    def __init__(self, in_channels, out_channels,
                 dropout = nn.Dropout3d, dropout_p = 0.,
                 conv = nn.Conv3d, conv_kwargs = None,
                 non_lin = nn.LeakyReLU, non_lin_kwargs = None,
                 norm = nn.BatchNorm3d, norm_kwargs = None,):

        """ From nnUnet """
        super(ConvDropoutNormNonlin, self).__init__()

        if non_lin_kwargs is None:
            non_lin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if norm_kwargs is None:
            norm_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dropout = dropout
        self.dropout_p = dropout_p
        self.non_lin = non_lin
        self.non_lin_op = self.non_lin(**non_lin_kwargs)

        self.norm = norm
        if self.norm is not None:
            self.norm_op = self.norm(out_channels, **norm_kwargs)
        else:
            self.norm_op = None

        self.conv = conv
        self.conv_op = self.conv(in_channels, out_channels, **conv_kwargs)

        if dropout_p > 0.:
            self.dropout_op = self.dropout(dropout_p, inplace = True)
        else:
            self.dropout_op = None

    def forward(self, x):
        # print(x.shape)
        x = self.conv_op(x)
        if self.dropout_op is not None:
            x = self.dropout_op(x)

        if self.norm is not None:
            x = self.norm_op(x)

        return self.non_lin_op(x)

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


class SoftDiceLoss(nn.Module):
    # {'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}
    def __init__(self, smooth=1e-5, batch_dice = True):
        """
        """
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.batch_dice = batch_dice

    def forward(self, ps, gt ):

        if self.batch_dice:
            dim = ()
        else:
            dim = (2,3,4)

        out = torch.sum(2 * ps * gt + self.smooth,dim=dim)
        out /= (torch.sum(gt,dim=dim) + torch.sum(ps,dim = dim) + self.smooth)
        out = out.mean() # In case of non-batch dice, it's the mean dice over images

        return -out


class DC_and_BCE_loss(nn.Module):
    def __init__(self, reduction = "sum", bce_kwargs = None, soft_dice_kwargs = None):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param reduction: "sum" only implemented
        """
        super(DC_and_BCE_loss, self).__init__()
        if bce_kwargs is None:
            bce_kwargs = {'reduction':reduction}
        if soft_dice_kwargs is None:
            soft_dice_kwargs = {}

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = SoftDiceLoss( **soft_dice_kwargs)

    def forward(self, net_output, target):
        ce_loss = self.ce(net_output, target)
        dc_loss = self.dc(torch.sigmoid(net_output), target)

        # print(ce_loss)
        # print(dc_loss)
        return ce_loss + dc_loss


class MultipleOutputLoss(nn.Module):
    def __init__(self, loss, weight_factors=None, interpolation = 'trilinear'):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss
        self.interpolation = interpolation

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        # Start with the last -> Assumes that this should be the final output so not interpolation
        l = self.loss(x[-1], y) * weights[-1]
        for i in range(0,len(x)-1):
            if weights[i] != 0:
                # Interpolate other outputs to label shape
                l += weights[i] * self.loss( F.interpolate( x[i], size = y.shape[2:],
                                                            mode = self.interpolation, align_corners = False ), y )
        return l


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

