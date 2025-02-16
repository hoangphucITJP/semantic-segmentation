"""
Code Adapted from:
https://github.com/sthalles/deeplab_v3

Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import torch
from torch import nn
from torch.nn import Sequential
from torchvision.ops.misc import ConvNormActivation

from .mynn import initialize_weights, Norm2d, Upsample
from .utils import get_aspp, get_trunk, make_seg_head


class DeepV3Plus(nn.Module):
    """
    DeepLabV3+ with various trunks supported
    Always stride8
    """

    def __init__(self, num_classes=1, trunk='wrn38',
                 use_dpc=False, init_all=False, input_channels=3):
        super(DeepV3Plus, self).__init__()
        self.backbone, s2_ch, _, high_level_ch = get_trunk(trunk, input_channels=input_channels)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                          bottleneck_ch=8,
                                          output_stride=8,
                                          dpc=use_dpc)
        self.bot_fine = nn.Conv2d(s2_ch, 8, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(aspp_out_ch, 8, kernel_size=1, bias=False)
        self.final = nn.Sequential(
            nn.Conv2d(8 + 8, 32, kernel_size=3, padding=1, bias=False),
            Norm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1, bias=True))

        if init_all:
            initialize_weights(self.aspp)
            initialize_weights(self.bot_aspp)
            initialize_weights(self.bot_fine)

        initialize_weights(self.final)

        if False:
            with torch.no_grad():
                self.final[-1].weight = torch.nn.parameter.Parameter(
                    torch.clip(self.final[-1].weight, max=0)
                )

        self.global_context_conv = DeepV3Plus.build_global_context_net()
        self.global_context_fc = nn.Linear(288, 16)

    @staticmethod
    def build_global_context_net():
        layers = [
            ConvNormActivation(16, 32, stride=2),
            nn.MaxPool2d(3, stride=2),
            ConvNormActivation(32, 64, stride=2),
            nn.AdaptiveAvgPool2d(1),
        ]
        return Sequential(*layers)

    def forward(self, inputs, noise_std=0):
        assert 'images' in inputs
        x = inputs['images']

        x_size = x.size()
        s2_features, final_features = self.backbone(x)
        aspp = self.aspp(final_features)
        conv_aspp = self.bot_aspp(aspp)
        conv_s2 = self.bot_fine(s2_features)
        conv_aspp = Upsample(conv_aspp, s2_features.size()[2:])
        cat_s4 = [conv_s2, conv_aspp]
        cat_s4 = torch.cat(cat_s4, 1)
        global_context_input = torch.squeeze(torch.squeeze(self.global_context_conv(final_features), 2), 2)
        global_context_input = torch.cat((global_context_input, inputs['black_white']), axis=1)
        global_context = self.global_context_fc(global_context_input)
        global_context = torch.unsqueeze(torch.unsqueeze(global_context, 2), 3)
        final = self.final(cat_s4 + global_context)
        up_sampled = Upsample(final, x_size[2:])

        mask = torch.sigmoid(up_sampled) + torch.normal(mean=0, std=noise_std, size=(1,)).to(up_sampled.device)
        mask = mask.clip(0, 1)
        cropped_mask = (x.mean(1, keepdims=True) > 0) * mask
        prediction = cropped_mask.amax((1, 2, 3))
        return {'mask': cropped_mask, 'prediction': prediction}


def DeepV3PlusSRNX50(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='seresnext-50', criterion=criterion)


def DeepV3PlusR50(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion)


def DeepV3PlusSRNX101(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='seresnext-101', criterion=criterion)


def DeepV3PlusW38(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='wrn38', criterion=criterion)


def DeepV3PlusW38I(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='wrn38', criterion=criterion,
                      init_all=True)


def DeepV3PlusX71(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='xception71', criterion=criterion)


def DeepV3PlusEffB4(num_classes, criterion):
    return DeepV3Plus(num_classes, trunk='efficientnet_b4',
                      criterion=criterion)


class DeepV3(nn.Module):
    """
    DeepLabV3 with various trunks supported
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None,
                 use_dpc=False, init_all=False, output_stride=8):
        super(DeepV3, self).__init__()
        self.criterion = criterion

        self.backbone, _s2_ch, _s4_ch, high_level_ch = \
            get_trunk(trunk, output_stride=output_stride)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                          bottleneck_ch=256,
                                          output_stride=output_stride,
                                          dpc=use_dpc)
        self.final = make_seg_head(in_ch=aspp_out_ch, out_ch=num_classes)

        initialize_weights(self.aspp)
        initialize_weights(self.final)

    def forward(self, inputs):
        assert 'images' in inputs
        x = inputs['images']

        x_size = x.size()
        _, _, final_features = self.backbone(x)
        aspp = self.aspp(final_features)
        final = self.final(aspp)
        out = Upsample(final, x_size[2:])

        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            return self.criterion(out, gts)

        return {'pred': out}


def DeepV3R50(num_classes, criterion):
    return DeepV3(num_classes, trunk='resnet-50', criterion=criterion)
