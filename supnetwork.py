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
        if config["pool"]:
            pool = set_layer_from_config(
                config['pool'])
        else: 
            pool = None
        if config["upconv"]:
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
        for block_config in config['arch_config']["blocks"]:
            b = UnetSkipConnectionBlock.build_from_config(block_config, submodule)
            submodule = b
            blocks.append(b)

        return SupUNet(blocks)

if __name__ == '__main__':
    for i in range(300):
        config = json.load(open("./models/configs/sample{}.json".format(i), 'r'))
        try:
            net = SupUNet.build_from_config(config)
            print(i)
        except: 
            print("non correct", i)