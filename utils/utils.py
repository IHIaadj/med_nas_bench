"""
Various utility methods in this module
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from PIL import Image

# Tell Matplotlib to not try and use interactive backend
mpl.use("agg")
import os
import sys

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

import torch
import torch.nn as nn
import torch.optim


def download_url(url, model_dir="~/.torch/proxyless_nas", overwrite=False):
    model_dir = os.path.expanduser(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file) or overwrite:
        os.makedirs(model_dir, exist_ok=True)
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return cached_file


def load_url(url, model_dir='~/.torch/proxyless_nas', map_location=None):
    cached_file = download_url(url, model_dir)
    map_location = "cpu" if not torch.cuda.is_available() and map_location is None else None
    return torch.load(cached_file, map_location=map_location)


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    logsoftmax = nn.LogSoftmax()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * \
        (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(
        kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def shuffle_layer(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # transpose
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


def get_split_list(in_dim, child_num):
    in_dim_list = [in_dim // child_num] * child_num
    for _i in range(in_dim % child_num):
        in_dim_list[_i] += 1
    return in_dim_list


def list_sum(x):
    if len(x) == 1:
        return x[0]
    else:
        return x[0] + list_sum(x[1:])


def count_parameters(model):
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    return total_params


def count_conv_flop(layer, x):
    out_h = int(x.size()[2] / layer.stride[0])
    out_w = int(x.size()[3] / layer.stride[1])
    delta_ops = layer.in_channels * layer.out_channels * \
        layer.kernel_size[0] * layer.kernel_size[1] * out_h * out_w / layer.groups
    return delta_ops


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BasicUnit(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    @property
    def unit_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError

def mpl_image_grid(images):
    """
    Create an image grid from an array of images. Show up to 16 images in one figure
    Arguments:
        image {Torch tensor} -- NxWxH array of images
    Returns:
        Matplotlib figure
    """
    # Create a figure to contain the plot.
    n = min(images.shape[0], 16) # no more than 16 thumbnails
    rows = 4
    cols = (n // 4) + (1 if (n % 4) != 0 else 0)
    figure = plt.figure(figsize=(2*rows, 2*cols))
    plt.subplots_adjust(0, 0, 1, 1, 0.001, 0.001)
    for i in range(n):
        # Start next subplot.
        plt.subplot(cols, rows, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if images.shape[1] == 3:
            # this is specifically for 3 softmax'd classes with 0 being bg
            # We are building a probability map from our three classes using 
            # fractional probabilities contained in the mask
            vol = images[i].detach().numpy()
            img = [[[(1-vol[0,x,y])*vol[1,x,y], (1-vol[0,x,y])*vol[2,x,y], 0] \
                            for y in range(vol.shape[2])] \
                            for x in range(vol.shape[1])]
            plt.imshow(img)
        else: # plotting only 1st channel
            plt.imshow((images[i, 0]*255).int(), cmap= "gray")

    return figure

def log_to_tensorboard(writer, loss, data, target, prediction_softmax, prediction, counter):
    """Logs data to Tensorboard
    Arguments:
        writer {SummaryWriter} -- PyTorch Tensorboard wrapper to use for logging
        loss {float} -- loss
        data {tensor} -- image data
        target {tensor} -- ground truth label
        prediction_softmax {tensor} -- softmax'd prediction
        prediction {tensor} -- raw prediction (to be used in argmax)
        counter {int} -- batch and epoch counter
    """
    writer.add_scalar("Loss",\
                    loss, counter)
    writer.add_figure("Image Data",\
        mpl_image_grid(data.float().cpu()), global_step=counter)
    writer.add_figure("Mask",\
        mpl_image_grid(target.float().cpu()), global_step=counter)
    writer.add_figure("Probability map",\
        mpl_image_grid(prediction_softmax.cpu()), global_step=counter)
    writer.add_figure("Prediction",\
        mpl_image_grid(torch.argmax(prediction.cpu(), dim=1, keepdim=True)), global_step=counter)

def save_numpy_as_image(arr, path):
    """
    This saves image (2D array) as a file using matplotlib
    Arguments:
        arr {array} -- 2D array of pixels
        path {string} -- path to file
    """
    plt.imshow(arr, cmap="gray") #Needs to be in row,col order
    plt.savefig(path)

def med_reshape(image, new_shape):
    """
    This function reshapes 3D data to new dimension padding with zeros
    and leaving the content in the top-left corner
    Arguments:
        image {array} -- 3D array of pixel data
        new_shape {3-tuple} -- expected output shape
    Returns:
        3D array of desired shape, padded with zeroes
    """

    reshaped_image = np.zeros(new_shape)

    reshaped_image[:image.shape[0], :image.shape[1], :image.shape[2]] = image

    return reshaped_image