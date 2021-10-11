"""
Network Initializations
"""

import importlib

import torch
from runx.logx import logx


def get_net(args, input_channels=3, backbone='wrn38'):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(network='module.semantic_segmentation.network.' + args.arch,
                    input_channels=input_channels, backbone=backbone)
    num_params = sum([param.nelement() for param in net.parameters()])
    logx.msg('Model params = {:2.1f}M'.format(num_params / 1000000))

    return net


def is_gscnn_arch(args):
    """
    Network is a GSCNN network
    """
    return 'gscnn' in args.arch


def wrap_network_in_dataparallel(net, use_apex_data_parallel=False):
    """
    Wrap the network in Dataparallel
    """
    if use_apex_data_parallel:
        import apex
        net = apex.parallel.DistributedDataParallel(net)
    else:
        net = torch.nn.DataParallel(net)
    return net


def get_model(network, input_channels=3, backbone='wrn38'):
    """
    Fetch Network Function Pointer
    """
    module = network[:network.rfind('.')]
    model = network[network.rfind('.') + 1:]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(input_channels=input_channels, trunk=backbone)
    return net
