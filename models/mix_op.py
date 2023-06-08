import numpy as np

from torch.nn.parameter import Parameter
import torch.nn.functional as F

from layers import *


def build_candidate_ops(candidate_ops, in_channels, out_channels, stride, ops_order):
    if candidate_ops is None:
        raise ValueError('please specify a candidate set')

    name2ops = {
        'Identity': lambda in_C, out_C, S: IdentityLayer(in_C, out_C, ops_order=ops_order),
        'Zero': lambda in_C, out_C, S: ZeroLayer(stride=S),
    }
    # add Contract layers. 
    # Each contract layer has its oposite expand layer automatically generated in UNetRecursiveblock.
    name2ops.update({
        '3x3_1_bn_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 3, S, 1, use_bn=True, act_func="relu"),
        '5x5_1_bn_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 5, S, 1, use_bn=True, act_func="relu"),
        '7x7_1_bn_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 7, S, 1, use_bn=True, act_func="relu"),
        '3x3_1_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 3, S, 1, use_bn=False, act_func="relu"),
        '5x5_1_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 5, S, 1, use_bn=False, act_func="relu"),
        '7x7_1_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 7, S, 1, use_bn=False, act_func="relu"),
        ####################################################
        '3x3_6_bn_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 3, S, 1, use_bn=True, act_func="relu6"),
        '5x5_6_bn_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 5, S, 1, use_bn=True, act_func="relu6"),
        '7x7_6_bn_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 7, S, 1, use_bn=True, act_func="relu6"),
        '3x3_6_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 3, S, 1, use_bn=False, act_func="relu6"),
        '5x5_6_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 5, S, 1, use_bn=False, act_func="relu6"),
        '7x7_6_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 7, S, 1, use_bn=False, act_func="relu6"),
        ####################################
        '3x3_l_bn_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 3, S, 1, use_bn=True, act_func="LeakyRelu"),
        '5x5_l_bn_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 5, S, 1, use_bn=True, act_func="LeakyRelu"),
        '7x7_l_bn_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 7, S, 1, use_bn=True, act_func="LeakyRelu"),
        '3x3_l_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 3, S, 1, use_bn=False, act_func="LeakyRelu"),
        '5x5_l_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 5, S, 1, use_bn=False, act_func="LeakyRelu"),
        '7x7_l_Contract': lambda in_C, out_C, S: Contract(in_C, out_C, 7, S, 1, use_bn=False, act_func="LeakyRelu")
    })

    return [
        name2ops[name](in_channels, out_channels, stride) for name in candidate_ops
    ]
