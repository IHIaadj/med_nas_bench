import math
from models.layers import *
import json

class UnetSkipConnectionBlock(BasicUnit):
    def __init__(self, pool, conv1, conv2, conv3, conv4, upconv, type="standard", submodule=None):
        super(UnetSkipConnectionBlock, self).__init__()
        self.type = type
        self.pool = pool
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.upconv = upconv
        self.submodule = submodule

        if type == "inner":
            model = [pool, conv1, conv2, upconv]
        elif type == "outer":
            model = [conv1, conv2, submodule, conv3, conv4, upconv]
        else:
            model = [pool, conv1, conv2, submodule, conv3, conv4, upconv]

        self.model = nn.Sequential(*model)

    @property
    def config(self):
        return {
            'name': UnetSkipConnectionBlock.__name__,
            'type': self.type,
            'conv1': self.conv1.config,
            'conv2': self.conv2.config,
            'conv3': self.conv3.config,
            'conv4': self.conv4.config,
            'pool': self.pool.config,
            'upconv': self.upconv.config
        }

    def add_submodule(self, submodule): 
        self.submodule = submodule
        if self.type == "inner":
            model = [self.pool, self.conv1, self.conv2, self.upconv]
        elif type == "outer":
            model = [self.conv1, self.conv2, submodule, self.conv3, self.conv4, self.upconv]
        else:
            model = [self.pool, self.conv1, self.conv2, submodule, self.conv3, self.conv4, self.upconv]

        self.model = nn.Sequential(*model)

    @staticmethod
    def build_from_config(config, submodule):
        if config['conv1']:
            conv1 = set_layer_from_config(
            config['conv1'])
        else:
            conv1 = None
        if config['conv2']:
            conv2 = set_layer_from_config(
            config['conv2'])
        else:
            conv2 = None      
        if config['conv3']:
            conv3 = set_layer_from_config(
            config['conv3'])
        else:
            conv3 = None
        if config['conv4']:
            conv4 = set_layer_from_config(
            config['conv4'])
        else:
            conv4 = None
        if config['pool']:
            pool = set_layer_from_config(
                config['pool'])
        else:
            pool = None

        if config['upconv']:
            upconv = set_layer_from_config(
                config['upconv'])
        else:
            upconv = None
        type = config["type"]
        return UnetSkipConnectionBlock(pool, conv1, conv2, conv3, conv4, upconv, type, submodule=submodule)

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x):
        if self.type == "outer":
            return self.model(x)
        else:
            crop = self.center_crop(self.model(x), x.size()[2], x.size()[3])
            return torch.cat([x, crop], 1)

"""
class UnetSkipConnectionBlock(BasicUnit):
    def __init__(self, in_channels=None, out_channels=None, num_classes=1, kernel_size=3,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        # downconv
        pool = nn.MaxPool2d(2, stride=2)
        conv1 = self.contract(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, norm_layer=norm_layer)
        conv2 = self.contract(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, norm_layer=norm_layer)

        # upconv
        conv3 = self.expand(in_channels=out_channels*2, out_channels=out_channels, kernel_size=kernel_size)
        conv4 = self.expand(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)

        if outermost:
            final = nn.Conv2d(out_channels, num_classes, kernel_size=1)
            down = [conv1, conv2]
            up = [conv3, conv4, final]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(in_channels*2, in_channels,
                                        kernel_size=2, stride=2)
            model = [pool, conv1, conv2, upconv]
        else:
            upconv = nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=2, stride=2)

            down = [pool, conv1, conv2]
            up = [conv3, conv4, upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, norm_layer=nn.InstanceNorm2d):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            norm_layer(out_channels),
            nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            crop = self.center_crop(self.model(x), x.size()[2], x.size()[3])
            return torch.cat([x, crop], 1)
"""
class SupUNet(nn.Module):
    def __init__(self, blocks):
        super(SupUNet, self).__init__()
        self.blocks = blocks[-1]
    def forward(self, x):
        return self.blocks(x)

    @property
    def unit_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': SupUNet.__name__,
            'blocks': [
                block.config for block in self.blocks],
        }

    @staticmethod
    def build_from_config(config):
        blocks = []
        submodule = None
        for block_config in config['blocks']:
            b = UnetSkipConnectionBlock.build_from_config(block_config, submodule)
            submodule = b
            blocks.append(b)

        return SupUNet(blocks)

if __name__ == '__main__':
    config = json.load(open("./config.config", 'r'))
    net = SupUNet.build_from_config(config)
    print(net)