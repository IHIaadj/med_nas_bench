# recursive implementation of Unet
import torch
import os
from torch import nn
import json
import nibabel as nib
from medpy.io import load
from utils import *
from layers import * 

class RUNet(nn.Module):
    def __init__(self, 
            num_classes=3, 
            in_channels=1, 
            initial_filter_size=64, 
            kernel_size=[[3, 3, 3, 3],[3, 3, 3, 3], [3, 3, 3, 3]],
            relu=[["relu"]*4]*3, 
            num_downs=4, 
            norm_layer=nn.InstanceNorm2d, 
            is_norm=[True]*3, 
            pool=[True]*3, 
            pool_type = ["max"]*3,
            dilation=[[1]*4]*3,
            order=[[1]*4]*3):
        # norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(RUNet, self).__init__()
        self.config = {}
        
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-1), 
                                             out_channels=initial_filter_size * 2 ** num_downs,
                                             num_classes=num_classes, 
                                             kernel_size=kernel_size[0], 
                                             norm_layer=norm_layer, 
                                             innermost=True, 
                                             relu=relu[0], 
                                             is_norm=is_norm[0], 
                                             pool=pool[0], 
                                             pool_type=pool_type[0],
                                             dilation=dilation[0],
                                             order=order[0])
        self.config["blocks"] = []
        self.config["blocks"].append(unet_block.to_json())
        for i in range(1, num_downs-1):
            unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-(i+1)),
                                                 out_channels=initial_filter_size * 2 ** (num_downs-i),
                                                 num_classes=num_classes, 
                                                 kernel_size=kernel_size[i], 
                                                 submodule=unet_block, 
                                                 norm_layer=norm_layer, 
                                                 relu=relu[i], 
                                                 is_norm=is_norm[i],
                                                 pool=pool[i],
                                                 pool_type=pool_type[i], 
                                                 dilation=dilation[i],
                                                 order=order[i])
            self.config["blocks"].append(unet_block.to_json())
        i = num_downs-1
        unet_block = UnetSkipConnectionBlock(in_channels=in_channels,
                                                 out_channels=initial_filter_size,
                                                 num_classes=num_classes, 
                                                 kernel_size=kernel_size[i], 
                                                 submodule=unet_block, 
                                                 norm_layer=norm_layer, 
                                                 outermost=True,
                                                 relu=relu[i], 
                                                 is_norm=is_norm[i],
                                                 pool=pool[i],
                                                 pool_type=pool_type[i], 
                                                 dilation=dilation[i],
                                                 order=order[i])

        self.config["blocks"].append(unet_block.to_json())
        self.model = unet_block

    def generate_config(self):
        return self.config

    def forward(self, x):
        return self.model(x)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, 
                in_channels=None, 
                out_channels=None, 
                num_classes=1, 
                kernel_size=[3, 3, 3, 3],
                submodule=None, 
                outermost=False, 
                innermost=False, 
                norm_layer=nn.InstanceNorm2d, 
                use_dropout=False, 
                relu=["relu"]*4, 
                pool=True, 
                pool_type="max",
                dilation=[1]*4,
                is_norm=True, 
                order=[1]*4):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = relu
        self.is_norm =is_norm
        self.order = order 

        # pooling
        if pool == True:
            self.pool = PoolingLayer(in_channels=in_channels, out_channels=out_channels, pool_type=pool_type)
        else:
            self.pool= None

        # down convs
        self.conv1 = self.contract(in_channels=in_channels, 
                                   out_channels=out_channels, 
                                   kernel_size=kernel_size[0], 
                                   norm_layer=norm_layer, 
                                   relu=relu[0], 
                                   norm=is_norm, 
                                   dilation=dilation[0],
                                   order=order[0])
        self.conv2 = self.contract(in_channels=out_channels, 
                                   out_channels=out_channels, 
                                   kernel_size=kernel_size[1], 
                                   norm_layer=norm_layer, 
                                   relu=relu[1], 
                                   norm=is_norm, 
                                   dilation=dilation[1],
                                   order=order[1])

        # upconv
        self.conv3 = self.expand(in_channels=out_channels*2, 
                                 out_channels=out_channels, 
                                 kernel_size=kernel_size[2], 
                                 relu=relu[2], 
                                 dilation=dilation[2],
                                 order=order[2])
        self.conv4 = self.expand(in_channels=out_channels, 
                                 out_channels=out_channels, 
                                 kernel_size=kernel_size[3], 
                                 relu=relu[3], 
                                 dilation=dilation[2],
                                 order=order[3])

        if outermost:
            self.final = nn.Conv2d(out_channels, num_classes, kernel_size=1)
            down = [self.conv1, self.conv2]
            up = [self.conv3, self.conv4, self.final]
            model = down + [submodule] + up
        elif innermost:
            self.upconv = nn.ConvTranspose2d(in_channels*2, in_channels,
                                        kernel_size=2, stride=2)
            model = [self.pool, self.conv1, self.conv2, self.upconv]
        else:
            self.upconv = nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=2, stride=2)

            down = [self.pool, self.conv1, self.conv2]
            up = [self.conv3, self.conv4, self.upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        model = [i for i in model if i is not None]
        self.model = nn.Sequential(*model)

    @staticmethod
    def generate_ordered_bn(weight, bn,bn2, act, order):
        if order == 1:
            layer = nn.Sequential((OrderedDict([
                ('conv', weight), 
                ('bn', bn), 
                ('act', act)])))
        elif order == 2: 
            layer = nn.Sequential((OrderedDict([
                 ('act', act),
                ('conv', weight), 
                ('bn', bn)])))
        else:
            layer = nn.Sequential((OrderedDict([
                ('bn', bn2),
                ('act', act),
                ('conv', weight)])))
        return layer 

    @staticmethod
    def generate_ordered(weight, act, order):
        if order == 1:
            layer = nn.Sequential((OrderedDict([
                ('conv', weight),
                ('act', act),
                ])))
        elif order == 2: 
            layer = nn.Sequential((OrderedDict([
                ('act', act),
                ('conv', weight)
                ])))
        else:
            layer = nn.Sequential((OrderedDict([
                ('act', act),
                ('conv', weight)
                ])))
        return layer 

    @staticmethod
    def contract(in_channels, out_channels, kernel_size, norm_layer=nn.InstanceNorm2d, relu="relu", norm=True, dilation= 1, order=1):
        if norm:
            if relu =="relu":
                weight = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, dilation=dilation)
                bn = norm_layer(out_channels)
                bn2 = norm_layer(in_channels)
                act = nn.ReLU(inplace=True)
                layer = UnetSkipConnectionBlock.generate_ordered_bn(weight, bn, bn2, act, order)
            elif relu == "relu6":
                weight = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, dilation=dilation)
                bn = norm_layer(out_channels)
                bn2 = norm_layer(in_channels)
                act = nn.ReLU6(inplace=True)
                layer = UnetSkipConnectionBlock.generate_ordered_bn(weight, bn, bn2, act, order)   
            else:
                weight = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, dilation=dilation)
                bn = norm_layer(out_channels)
                bn2 = norm_layer(in_channels)
                act = nn.LeakyReLU(inplace=True)
                layer = UnetSkipConnectionBlock.generate_ordered_bn(weight, bn, bn2, act, order)   
        else:
            if relu =="relu":
                weight = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, dilation=dilation)
                act = nn.ReLU(inplace=True)
                layer = UnetSkipConnectionBlock.generate_ordered(weight, act, order)  
            elif relu == "relu6":
                weight = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, dilation=dilation)
                act = nn.ReLU6(inplace=True)
                layer = UnetSkipConnectionBlock.generate_ordered(weight, act, order)  
            else:
                weight = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, dilation=dilation)
                act = nn.LeakyReLU(inplace=True)
                layer = UnetSkipConnectionBlock.generate_ordered(weight, act, order)  
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size, relu="relu", dilation=1, order=1):
        if relu =="relu":
            if order == 1:
                layer = nn.Sequential((OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, dilation=dilation)),
                ('act', nn.ReLU(inplace=True))
                ])))
            else: 
                layer = nn.Sequential((OrderedDict([
                ('act', nn.ReLU(inplace=True)),
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, dilation=dilation))
                ])))
        elif relu == "relu6":
            if order == 1:
                layer = nn.Sequential((OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, dilation=dilation)),
                ('act', nn.ReLU6(inplace=True))
                ])))
            else: 
                layer = nn.Sequential((OrderedDict([
                ('act', nn.ReLU6(inplace=True)),
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, dilation=dilation))
                ])))
        else:
            if order == 1:
                layer = nn.Sequential((OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, dilation=dilation)),
                ('act', nn.LeakyReLU(inplace=True))
                ])))
            else: 
                layer = nn.Sequential((OrderedDict([
                ('act', nn.LeakyReLU(inplace=True)),
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, dilation=dilation))
                ])))
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def to_json(self):
        config = {}
        config["name"] = "UnetSkipConnectionBlock"
        if self.outermost:
            config["type"] = "outer"
            config["pool"] = {}
            config["conv1"] = {}
            config["conv1"]["name"] = "Contract"
            config["conv1"]["in_channels"] = self.conv1.conv.in_channels
            config["conv1"]["out_channels"] = self.conv1.conv.out_channels
            config["conv1"]["kernel_size"] = self.conv1.conv.kernel_size[0]
            config["conv1"]["stride"] = self.conv1.conv.stride[0]
            config["conv1"]["dilation"] = self.conv1.conv.dilation
            config["conv1"]["use_bn"] = self.is_norm
            config["conv1"]["act_func"] = self.relu[0]
            config["conv1"]["order"] = self.order[0]

            config["conv2"] = {}
            config["conv2"]["name"] = "Contract"
            config["conv2"]["in_channels"] = self.conv2.conv.in_channels
            config["conv2"]["out_channels"] = self.conv2.conv.out_channels
            config["conv2"]["kernel_size"] = self.conv2.conv.kernel_size[0]
            config["conv2"]["stride"] = self.conv2.conv.stride[0]
            config["conv2"]["dilation"] = self.conv2.conv.dilation
            config["conv2"]["use_bn"] = self.is_norm
            config["conv2"]["act_func"] = self.relu[1]
            config["conv2"]["order"] = self.order[1]


            config["conv3"] = {}
            config["conv3"]["name"] = "Expand"
            config["conv3"]["in_channels"] = self.conv3.conv.in_channels
            config["conv3"]["out_channels"] = self.conv3.conv.out_channels
            config["conv3"]["kernel_size"] = self.conv3.conv.kernel_size[0]
            config["conv3"]["stride"] = self.conv3.conv.stride[0]
            config["conv3"]["dilation"] = self.conv3.conv.dilation
            config["conv3"]["act_func"] = self.relu[2]
            config["conv3"]["order"] = self.order[2]

            config["conv4"] = {}
            config["conv4"]["name"] = "Expand"
            config["conv4"]["in_channels"] = self.conv4.conv.in_channels
            config["conv4"]["out_channels"] = self.conv4.conv.out_channels
            config["conv4"]["kernel_size"] = self.conv4.conv.kernel_size[0]
            config["conv4"]["stride"] = self.conv4.conv.stride[0]
            config["conv4"]["dilation"] = self.conv4.conv.dilation
            config["conv4"]["act_func"] = self.relu[3]
            config["conv4"]["order"] = self.order[3]


            config["upconv"] = {}
            config["upconv"]["name"] = "Contract"
            config["upconv"]["in_channels"] = self.final.in_channels
            config["upconv"]["out_channels"] = self.final.out_channels
            config["upconv"]["kernel_size"] = self.final.kernel_size[0]
            config["upconv"]["stride"] = self.final.stride[0]

        elif self.innermost:
            config["type"] = "inner"
            config["pool"] = {}
            if self.pool:
                config["pool"]["name"] = "PoolingLayer"
                config["pool"]["in_channels"] = self.in_channels
                config["pool"]["out_channels"] = self.out_channels
                if isinstance(self.pool, nn.MaxPool2d):
                    config["pool"]["pool_type"] = "max"
                else:
                    config["pool"]["pool_type"] = "avg"
                config["pool"]["kernel_size"] = self.pool.kernel_size
                config["pool"]["stride"] = self.pool.stride

            config["conv1"] = {}
            config["conv1"]["name"] = "Contract"
            config["conv1"]["in_channels"] = self.conv1.conv.in_channels
            config["conv1"]["out_channels"] = self.conv1.conv.out_channels
            config["conv1"]["kernel_size"] = self.conv1.conv.kernel_size[0]
            config["conv1"]["stride"] = self.conv1.conv.stride[0]
            config["conv1"]["dilation"] = self.conv1.conv.dilation
            config["conv1"]["use_bn"] = self.is_norm
            config["conv1"]["act_func"] = self.relu[0]
            config["conv1"]["order"] = self.order[0]

            config["conv2"] = {}
            config["conv2"]["name"] = "Contract"
            config["conv2"]["in_channels"] = self.conv2.conv.in_channels
            config["conv2"]["out_channels"] = self.conv2.conv.out_channels
            config["conv2"]["kernel_size"] = self.conv2.conv.kernel_size[0]
            config["conv2"]["stride"] = self.conv2.conv.stride[0]
            config["conv2"]["dilation"] = self.conv2.conv.dilation
            config["conv2"]["use_bn"] = self.is_norm
            config["conv2"]["act_func"] = self.relu[1]
            config["conv2"]["order"] = self.order[1]

            config["conv3"] = {}

            config["conv4"] = {}

            config["upconv"] = {}
            config["upconv"]["name"] = "TransposeConv2D"
            config["upconv"]["in_channels"] = self.upconv.in_channels
            config["upconv"]["out_channels"] = self.upconv.out_channels
            config["upconv"]["kernel_size"] = self.upconv.kernel_size[0]
            config["upconv"]["stride"] = self.upconv.stride[0]
        else:
            config["type"] = "standard"
            config["pool"] = {}
            if self.pool:
                config["pool"]["name"] = "PoolingLayer"
                config["pool"]["in_channels"] = self.in_channels
                config["pool"]["out_channels"] = self.out_channels
                if isinstance(self.pool, nn.MaxPool2d):
                    config["pool"]["pool_type"] = "max"
                else:
                    config["pool"]["pool_type"] = "avg"
                config["pool"]["kernel_size"] = self.pool.kernel_size
                config["pool"]["stride"] = self.pool.stride

            config["conv1"] = {}
            config["conv1"]["name"] = "Contract"
            config["conv1"]["in_channels"] = self.conv1.conv.in_channels
            config["conv1"]["out_channels"] = self.conv1.conv.out_channels
            config["conv1"]["kernel_size"] = self.conv1.conv.kernel_size[0]
            config["conv1"]["stride"] = self.conv1.conv.stride[0]
            config["conv1"]["dilation"] = self.conv1.conv.dilation
            config["conv1"]["use_bn"] = self.is_norm
            config["conv1"]["act_func"] = self.relu[0]
            config["conv1"]["order"] = self.order[0]

            config["conv2"] = {}
            config["conv2"]["name"] = "Contract"
            config["conv2"]["in_channels"] = self.conv2.conv.in_channels
            config["conv2"]["out_channels"] = self.conv2.conv.out_channels
            config["conv2"]["kernel_size"] = self.conv2.conv.kernel_size[0]
            config["conv2"]["stride"] = self.conv2.conv.stride[0]
            config["conv2"]["dilation"] = self.conv2.conv.dilation
            config["conv2"]["use_bn"] = self.is_norm
            config["conv2"]["act_func"] = self.relu[1]
            config["conv2"]["order"] = self.order[1]

            config["conv3"] = {}
            config["conv3"]["name"] = "Expand"
            config["conv3"]["in_channels"] = self.conv3.conv.in_channels
            config["conv3"]["out_channels"] = self.conv3.conv.out_channels
            config["conv3"]["kernel_size"] = self.conv3.conv.kernel_size[0]
            config["conv3"]["stride"] = self.conv3.conv.stride[0]
            config["conv3"]["dilation"] = self.conv3.conv.dilation
            config["conv3"]["act_func"] = self.relu[2]
            config["conv3"]["order"] = self.order[2]

            config["conv4"] = {}
            config["conv4"]["name"] = "Expand"
            config["conv4"]["in_channels"] = self.conv4.conv.in_channels
            config["conv4"]["out_channels"] = self.conv4.conv.out_channels
            config["conv4"]["kernel_size"] = self.conv4.conv.kernel_size[0]
            config["conv4"]["stride"] = self.conv4.conv.stride[0]
            config["conv4"]["dilation"] = self.conv4.conv.dilation
            config["conv4"]["act_func"] = self.relu[3]
            config["conv4"]["order"] = self.order[3]

            config["upconv"] = {}
            config["upconv"]["name"] = "TransposeConv2D"
            config["upconv"]["in_channels"] = self.upconv.in_channels
            config["upconv"]["out_channels"] = self.upconv.out_channels
            config["upconv"]["kernel_size"] = self.upconv.kernel_size[0]
            config["upconv"]["stride"] = self.upconv.stride[0]

        return config

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            crop = self.center_crop(self.model(x), x.size()[2], x.size()[3])
            return torch.cat([x, crop], 1)

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def sample(name, init_filter, num_down, ks, is_norm, pool, pool_type, relu, dilation, order):
    try:
        model = RUNet( 
                      initial_filter_size=init_filter, 
                      kernel_size=ks, 
                      num_downs=num_down, 
                      is_norm=is_norm, 
                      pool=pool, 
                      pool_type= pool_type,
                      relu=relu,
                      dilation=dilation, 
                      order=order)
    except RuntimeError:
        print("memory problem")
        return 0, 0, 0

    
    image, _ = load("hippocampus_001.nii.gz")
    image = image / np.max(image)
    image = med_reshape(image, new_shape=(image.shape[0], 64, 64))
    try:
        #print(model)
        model.eval()
        model.float()
        _ = single_volume_inference(model, image)
        config = model.generate_config()
        param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size = get_model_size(model)
        del model 
        return config, param, model_size
    except Exception as e:
        print(e)
        return 0, 0, 0

import random 
import numpy as np
import pickle

def single_volume_inference(model, volume):
    """
    Runs inference on a single volume of conformant patch size
    Arguments:
        volume {Numpy array} -- 3D array representing the volume
    Returns:
        3D NumPy array with prediction mask
    """
    model.eval()

    # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
    slices = []

    with torch.no_grad():
        for idx0 in range(volume.shape[0]):
            slc = volume[idx0, :, :]
            slc = slc[None, None, :]
            slc_ts = torch.from_numpy(slc)
            slc_ts = torch.Tensor(slc_ts).float()
            #print(slc_ts)
            prediction = model(slc_ts)
            
            msk = prediction.argmax(axis=1).cpu().numpy()
            slices += [msk]
            
    return np.concatenate(slices, axis=0)
    
if __name__ == '__main__':
    configs = []
    nb = 0
    while nb < 1000:
        c = {}
        print(nb)
        #initial filter size 
        init_filter = random.choice([8, 16, 32, 48, 96, 128, 144, 169, 196, 224, 256])

        num_down = np.random.randint(1, 5)
        # kernel size
        ks = []
        t= random.choice([3, 5])
        ks = [[t]*4]*num_down

        # is norm 
        t = random.choices([True, False], k=num_down)
        is_norm = t 

        # pool 
        t = random.choices([True, False], k=num_down)
        pool = t 

        t = random.choices(["max", "avg"], k=num_down)
        pool_type = t 

        # relu 
        relu = []
        for i in range(num_down):
            t = random.choices(["relu", "relu6", "LeakyRelu"], k=4)
            relu.append(t) 

        # order 
        order = []
        for i in range(num_down):
            t = random.choices([1, 2, 3], k=4)
            order.append(t) 

        #dilation
        dilation = []
        for i in range(num_down):
            t = random.choices([1, 2], k=4)
            dilation.append(t) 

        dcs = random.random()*(0.95-0.5)+0.46
        jcs = random.random()*(0.95-0.5)+0.36
        sensitivity = random.random()*(0.95-0.5)+0.55
        specificity = random.random()*(0.95-0.5)+0.2
        metrics = {"dcs":dcs, "jcs": jcs, "sensitivity": sensitivity, "specificity": specificity}
        name = 'unet_like_model_{}'.format(nb)
        #print(init_filter, ks, num_down, is_norm, pool, relu)
        #print("\n")
        #print((name, init_filter, num_down, ks, is_norm, pool, pool_type, relu, dilation, order))
        config, param, model_size = sample(name, init_filter, num_down, ks, is_norm, pool, pool_type, relu, dilation, order)
        if config == None or config == 0:
            continue
        c["id"] = nb
        c["name"] = name
        c["metrics"] = metrics
        c["params"] = param
        c["model_size"] = param
        c["arch_config"] = config
        configs.append(c)
        # Serializing json
        json_object = json.dumps(c, indent=4)
 
        # Writing to sample.json
        with open("./configs/sample{}.json".format(nb), "w") as outfile:
            outfile.write(json_object)

        del json_object
        nb+=1
    print(configs)
    with open("test.pickle", 'wb') as pick:
        pickle.dump(configs, pick)
    
    
