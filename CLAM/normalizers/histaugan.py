import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import random
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

from normalizers.base import BaseNormalizer

class RandomRotate90:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


geom_augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    RandomRotate90([0, 90, 180, 270]),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomErasing(),
])

# geometric augmentations + brightness/contrast jitter + Gaussian blur + random erasing
basic_augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    RandomRotate90([0, 90, 180, 270]),
    transforms.RandomApply((transforms.GaussianBlur(3), ), p=0.25),
    transforms.RandomApply((transforms.ColorJitter(
        brightness=0.1, contrast=0.1), ), p=0.5),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomErasing(),
])

# same as geometric augmentations
gan_augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    RandomRotate90([0, 90, 180, 270]),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomErasing(),
])

# basic augmentations + hue/saturation jitter
color_augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    RandomRotate90([0, 90, 180, 270]),
    transforms.RandomApply((transforms.GaussianBlur(3), ), p=0.25),
    transforms.RandomApply((transforms.ColorJitter(
        brightness=0.1, contrast=0.1), ), p=0.5),
    transforms.RandomApply(
        (transforms.ColorJitter(saturation=0.5, hue=0.5), ), p=0.5),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomErasing(),
])

# basic augmentations + light hue/saturation jitter
color_augmentations_light = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    RandomRotate90([0, 90, 180, 270]),
    transforms.RandomApply((transforms.GaussianBlur(3), ), p=0.25),
    transforms.RandomApply((transforms.ColorJitter(
        brightness=0.1, contrast=0.1), ), p=0.5),
    transforms.RandomApply(
        (transforms.ColorJitter(saturation=0.1, hue=0.1), ), p=0.5),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomErasing(),
])

no_augmentations = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


def normalization(center):
    assert center in range(5), 'center not valid, should be in range(5)'
    mean = [
        [0.6710, 0.5327, 0.6448],
        [0.6475, 0.5139, 0.6222],
        [0.7875, 0.6251, 0.7567],
        [0.4120, 0.3270, 0.3959],
        [0.7324, 0.5814, 0.7038]
    ]
    std = [
        [0.2083, 0.2294, 0.1771],
        [0.2060, 0.2261, 0.1754],
        [0.2585, 0.2679, 0.2269],
        [0.2605, 0.2414, 0.2394],
        [0.2269, 0.2450, 0.1950]
    ]

    return mean[center], std[center]


class Args:
    concat = 1
    crop_size = 216  # only used as an argument for training
    dis_norm = None
    dis_scale = 3
    dis_spectral_norm = False
    dataroot = 'data'
    gpu = 1
    input_dim = 3
    nThreads = 4
    num_domains = 5
    nz = 8
    # resume = False
    resume = '/shared/cgorner/HistAuGAN/gan_weights.pth'


mean_domains = [
    torch.tensor([0.3020, -2.6476, -0.9849, -0.7820, -
                 0.2746,  0.3361,  0.1694, -1.2148]),
    torch.tensor([0.1453, -1.2400, -0.9484,  0.9697, -
                 2.0775,  0.7676, -0.5224, -0.2945]),
    torch.tensor([2.1067, -1.8572,  0.0055,  1.2214, -
                 2.9363,  2.0249, -0.4593, -0.9771]),
    torch.tensor([0.8378, -2.1174, -0.6531,  0.2986, -
                 1.3629, -0.1237, -0.3486, -1.0716]),
    torch.tensor([1.6073,  1.9633, -0.3130, -1.9242, -
                 0.9673,  2.4990, -2.2023, -1.4109]),
]

std_domains = [
    torch.tensor([0.6550, 1.5427, 0.5444, 0.7254,
                 0.6701, 1.0214, 0.6245, 0.6886]),
    torch.tensor([0.4143, 0.6543, 0.5891, 0.4592,
                 0.8944, 0.7046, 0.4441, 0.3668]),
    torch.tensor([0.5576, 0.7634, 0.7875, 0.5220,
                 0.7943, 0.8918, 0.6000, 0.5018]),
    torch.tensor([0.4157, 0.4104, 0.5158, 0.3498,
                 0.2365, 0.3612, 0.3375, 0.4214]),
    torch.tensor([0.6154, 0.3440, 0.7032, 0.6220,
                 0.4496, 0.6488, 0.4886, 0.2989]),
]


def generate_hist_augs(img, img_domain, model, z_content=None, same_attribute=False, new_domain=None, stats=None, device=torch.device('cpu')):
    """
    Generates a new stain color for the input image img.

    :img: input image of shape (3, 216, 216) [type: torch.Tensor]
    :img_domain: int in range(5)
    :model: HistAuGAN model
    :z_content: content encoding, if None this will be computed from img
    :same_attribute: [type: bool] indicates whether the attribute encoding of img or a randomly generated attribute are used
    :new_domain: either int in range(5) or torch.Tensor of shape (1, 5)
    :stats: (mean, std dev) of the latent space of HistAuGAN
    :device: torch.device to map the tensors to
    """
    # compute content vector
    if z_content is None:
        z_content = model.enc_c(img.sub(0.5).mul(2).unsqueeze(0))

    # compute attribute
    if same_attribute:
        mu, logvar = model.enc_a.forward(img.sub(0.5).mul(
            2).unsqueeze(0), torch.eye(5)[img_domain].unsqueeze(0).to(device))
        std = logvar.mul(0.5).exp_().to(device)
        eps = torch.randn((std.size(0), std.size(1))).to(device)
        z_attr = eps.mul(std).add_(mu)
    elif same_attribute == False and stats is not None and new_domain in range(5):
        z_attr = (torch.randn((1, 8, )) * \
            stats[1][new_domain] + stats[0][new_domain]).to(device)
    else:
        z_attr = torch.randn((1, 8, )).to(device)

    # determine new domain vector
    if isinstance(new_domain, int) and new_domain in range(5):
        new_domain = torch.eye(5)[new_domain].unsqueeze(0).to(device)
    elif isinstance(new_domain, torch.Tensor) and new_domain.shape == (1, 5):
        new_domain = new_domain.to(device)
    else:
        new_domain = torch.eye(5)[np.random.randint(5)].unsqueeze(0).to(device)

    # generate new histology image with same content as img
    out = model.gen(z_content, z_attr, new_domain).detach().squeeze(0)  # in range [-1, 1]

    return out

class MD_E_content(nn.Module):
    def __init__(self, input_dim):
        super(MD_E_content, self).__init__()
        enc_c = []
        tch = 64
        enc_c += [LeakyReLUConv2d(input_dim, tch,
                                  kernel_size=7, stride=1, padding=3)]
        for i in range(1, 3):
            enc_c += [ReLUINSConv2d(tch, tch * 2,
                                    kernel_size=3, stride=2, padding=1)]
            tch *= 2
        for i in range(0, 3):
            enc_c += [INSResBlock(tch, tch)]

        for i in range(0, 1):
            enc_c += [INSResBlock(tch, tch)]
            enc_c += [GaussianNoiseLayer()]
        self.conv = nn.Sequential(*enc_c)

    def forward(self, x):
        return self.conv(x)


class MD_E_attr(nn.Module):
    def __init__(self, input_dim, output_nc=8, c_dim=3):
        super(MD_E_attr, self).__init__()
        dim = 64
        self.model = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim+c_dim, dim, 7, 1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim*2, 4, 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim*2, dim*4, 4, 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim*4, dim*4, 4, 2),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim*4, dim*4, 4, 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim*4, output_nc, 1, 1, 0))

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x_c = torch.cat([x, c], dim=1)
        output = self.model(x_c)
        return output.view(output.size(0), -1)


class MD_E_attr_concat(nn.Module):
    def __init__(self, input_dim, output_nc=8, c_dim=3, norm_layer=None, nl_layer=None):
        super(MD_E_attr_concat, self).__init__()

        ndf = 64
        n_blocks = 4
        max_ndf = 4

        conv_layers = [nn.ReflectionPad2d(1)]
        conv_layers += [nn.Conv2d(input_dim+c_dim, ndf,
                                  kernel_size=4, stride=2, padding=0, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)  # 2**(n-1)
            output_ndf = ndf * min(max_ndf, n+1)  # 2**n
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AdaptiveAvgPool2d(1)]  # AvgPool2d(13)
        self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x_c = torch.cat([x, c], dim=1)
        x_conv = self.conv(x_c)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        outputVar = self.fcVar(conv_flat)
        return output, outputVar


class MD_G_uni(nn.Module):
    def __init__(self, output_dim, c_dim=3):
        super(MD_G_uni, self).__init__()
        self.c_dim = c_dim
        tch = 256
        dec_share = []
        dec_share += [INSResBlock(tch, tch)]
        self.dec_share = nn.Sequential(*dec_share)
        tch = 256+self.c_dim
        dec = []
        for i in range(0, 3):
            dec += [INSResBlock(tch, tch)]
        dec += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3,
                                       stride=2, padding=1, output_padding=1)]
        tch = tch//2
        dec += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3,
                                       stride=2, padding=1, output_padding=1)]
        tch = tch//2
        dec += [nn.ConvTranspose2d(tch, output_dim,
                                   kernel_size=1, stride=1, padding=0)]+[nn.Tanh()]
        self.dec = nn.Sequential(*dec)

    def forward(self, x, c):
        out0 = self.dec_share(x)
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, out0.size(2), out0.size(3))
        x_c = torch.cat([out0, c], dim=1)
        return self.dec(x_c)


class MD_G_multi_concat(nn.Module):
    def __init__(self, output_dim, c_dim=3, nz=8):
        super(MD_G_multi_concat, self).__init__()
        self.nz = nz
        self.c_dim = c_dim
        tch = 256
        dec_share = []
        dec_share += [INSResBlock(tch, tch)]
        self.dec_share = nn.Sequential(*dec_share)
        tch = 256+self.nz+self.c_dim
        dec1 = []
        for i in range(0, 3):
            dec1 += [INSResBlock(tch, tch)]
        tch = tch + self.nz
        dec2 = [ReLUINSConvTranspose2d(
            tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch = tch//2
        tch = tch + self.nz
        dec3 = [ReLUINSConvTranspose2d(
            tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
        tch = tch//2
        tch = tch + self.nz
        dec4 = [nn.ConvTranspose2d(
            tch, output_dim, kernel_size=1, stride=1, padding=0)]+[nn.Tanh()]
        self.dec1 = nn.Sequential(*dec1)
        self.dec2 = nn.Sequential(*dec2)
        self.dec3 = nn.Sequential(*dec3)
        self.dec4 = nn.Sequential(*dec4)

    def forward(self, x, z, c):
        out0 = self.dec_share(x)
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
            z.size(0), z.size(1), x.size(2), x.size(3))
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, out0.size(2), out0.size(3))
        x_c_z = torch.cat([out0, c, z_img], 1)
        out1 = self.dec1(x_c_z)
        z_img2 = z.view(z.size(0), z.size(1), 1, 1).expand(
            z.size(0), z.size(1), out1.size(2), out1.size(3))
        x_and_z2 = torch.cat([out1, z_img2], 1)
        out2 = self.dec2(x_and_z2)
        z_img3 = z.view(z.size(0), z.size(1), 1, 1).expand(
            z.size(0), z.size(1), out2.size(2), out2.size(3))
        x_and_z3 = torch.cat([out2, z_img3], 1)
        out3 = self.dec3(x_and_z3)
        z_img4 = z.view(z.size(0), z.size(1), 1, 1).expand(
            z.size(0), z.size(1), out3.size(2), out3.size(3))
        x_and_z4 = torch.cat([out3, z_img4], 1)
        out4 = self.dec4(x_and_z4)
        return out4


class MD_G_multi(nn.Module):
    def __init__(self, output_dim, c_dim=3, nz=8):
        super(MD_G_multi, self).__init__()
        self.nz = nz
        ini_tch = 256
        tch_add = ini_tch
        tch = ini_tch
        self.tch_add = tch_add
        self.dec1 = MisINSResBlock(tch, tch_add)
        self.dec2 = MisINSResBlock(tch, tch_add)
        self.dec3 = MisINSResBlock(tch, tch_add)
        self.dec4 = MisINSResBlock(tch, tch_add)

        dec5 = []
        dec5 += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3,
                                        stride=2, padding=1, output_padding=1)]
        tch = tch//2
        dec5 += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3,
                                        stride=2, padding=1, output_padding=1)]
        tch = tch//2
        dec5 += [nn.ConvTranspose2d(tch, output_dim,
                                    kernel_size=1, stride=1, padding=0)]
        dec5 += [nn.Tanh()]
        self.decA5 = nn.Sequential(*dec5)

        self.mlp = nn.Sequential(
            nn.Linear(nz+c_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, tch_add*4))
        return

    def forward(self, x, z, c):
        z_c = torch.cat([c, z], 1)
        z_c = self.mlp(z_c)
        z1, z2, z3, z4 = torch.split(z_c, self.tch_add, dim=1)
        z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()
        out1 = self.dec1(x, z1)
        out2 = self.dec2(out1, z2)
        out3 = self.dec3(out2, z3)
        out4 = self.dec4(out3, z4)
        out = self.decA5(out4)
        return out


class MD_Dis(nn.Module):
    def __init__(self, input_dim, norm='None', sn=False, c_dim=3, image_size=216):
        super(MD_Dis, self).__init__()
        ch = 64
        n_layer = 6
        self.model, curr_dim = self._make_net(ch, input_dim, n_layer, norm, sn)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=1,
                               stride=1, padding=1, bias=False)
        kernal_size = int(image_size/np.power(2, n_layer))
        self.conv2 = nn.Conv2d(
            curr_dim, c_dim, kernel_size=kernal_size, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3,
                                  stride=2, padding=1, norm=norm, sn=sn)]  # 16
        tch = ch
        for i in range(1, n_layer-1):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3,
                                      stride=2, padding=1, norm=norm, sn=sn)]  # 8
            tch *= 2
        model += [LeakyReLUConv2d(tch, tch, kernel_size=3,
                                  stride=2, padding=1, norm='None', sn=sn)]  # 2
        # model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)] # 2
        #tch *= 2
        return nn.Sequential(*model), tch

    def cuda(self, gpu):
        self.model.cuda(gpu)
        self.conv1.cuda(gpu)
        self.conv2.cuda(gpu)

    def forward(self, x):
        h = self.model(x)
        out = self.conv1(h)
        out_cls = self.conv2(h)
        out_cls = self.pool(out_cls)
        return out, out_cls.view(out_cls.size(0), out_cls.size(1))


class MD_Dis_content(nn.Module):
    def __init__(self, c_dim=3):
        super(MD_Dis_content, self).__init__()
        model = []
        model += [LeakyReLUConv2d(256, 256, kernel_size=7,
                                  stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(256, 256, kernel_size=7,
                                  stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(256, 256, kernel_size=7,
                                  stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(256, 256, kernel_size=4,
                                  stride=1, padding=0)]
        model += [nn.Conv2d(256, c_dim, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out = out.view(out.size(0), out.size(1))
        return out


####################################################################
# ------------------------- Basic Functions -------------------------
####################################################################
def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / \
                float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    else:
        return NotImplementedError('no such learn rate policy')
    return scheduler


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += conv3x3(inplanes, outplanes)
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % layer_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=False)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def conv3x3(in_planes, out_planes):
    return [nn.ReflectionPad2d(1), nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=True)]


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

####################################################################
# -------------------------- Basic Blocks --------------------------
####################################################################

# The code of LayerNorm is modified from MUNIT (https://github.com/NVlabs/MUNIT)
class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
        return

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(x, normalized_shape)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += conv3x3(inplanes, inplanes)
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [spectral_norm(nn.Conv2d(n_in, n_out,
                                    kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                                stride=stride, padding=0, bias=True)]
        if 'norm' == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.LeakyReLU(inplace=False)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        # elif == 'Group'

    def forward(self, x):
        return self.model(x)


class ReLUINSConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(ReLUINSConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                            stride=stride, padding=0, bias=True)]
        model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class INSResBlock(nn.Module):
    def conv3x3(self, inplanes, out_planes, stride=1):
        return [nn.ReflectionPad2d(1), nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]

    def __init__(self, inplanes, planes, stride=1, dropout=0.0):
        super(INSResBlock, self).__init__()
        model = []
        model += self.conv3x3(inplanes, planes, stride)
        model += [nn.InstanceNorm2d(planes)]
        model += [nn.ReLU(inplace=False)]
        model += self.conv3x3(planes, planes)
        model += [nn.InstanceNorm2d(planes)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class MisINSResBlock(nn.Module):
    def conv3x3(self, dim_in, dim_out, stride=1):
        return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))

    def conv1x1(self, dim_in, dim_out):
        return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

    def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
        super(MisINSResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm2d(dim))
        self.conv2 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm2d(dim))
        self.blk1 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False))
        self.blk2 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False))
        model = []
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.blk1.apply(gaussian_weights_init)
        self.blk2.apply(gaussian_weights_init)

    def forward(self, x, z):
        residual = x
        z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(
            z.size(0), z.size(1), x.size(2), x.size(3))
        o1 = self.conv1(x)
        o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
        o3 = self.conv2(o2)
        out = self.blk2(torch.cat([o3, z_expand], dim=1))
        out += residual
        return out


class GaussianNoiseLayer(nn.Module):
    def __init__(self,):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable(torch.randn(x.size())).to(device)
        return x + noise


class ReLUINSConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
        super(ReLUINSConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride,
                                     padding=padding, output_padding=output_padding, bias=True)]
        model += [LayerNorm(n_out)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


####################################################################
# --------------------- Spectral Normalization ---------------------
#  This part of code is copied from pytorch master branch (0.5.0)
####################################################################
class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.matmul(
                    weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.matmul(weight_mat, v),
                                dim=0, eps=self.eps)
        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        return weight, u

    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight))

    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)
        u = F.normalize(weight.new_empty(
            height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module, name='weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module
    raise ValueError(
        "spectral_norm of '{}' not found in {}".format(name, module))

class MD_multi(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        lr = 0.0001
        lr_dcontent = lr/2.5
        self.nz = 8

        if opts.concat == 1:
            self.concat = True
        else:
            self.concat = False

        self.dis1 = MD_Dis(opts.input_dim, norm=opts.dis_norm,
                                    sn=opts.dis_spectral_norm, c_dim=opts.num_domains, image_size=opts.crop_size)
        self.dis2 = MD_Dis(opts.input_dim, norm=opts.dis_norm,
                                    sn=opts.dis_spectral_norm, c_dim=opts.num_domains, image_size=opts.crop_size)
        self.enc_c = MD_E_content(opts.input_dim)
        if self.concat:
            self.enc_a = MD_E_attr_concat(opts.input_dim, output_nc=self.nz, c_dim=opts.num_domains,
                                                   norm_layer=None, nl_layer=get_non_linearity(layer_type='lrelu'))
            self.gen = MD_G_multi_concat(
                opts.input_dim, c_dim=opts.num_domains, nz=self.nz)
        else:
            self.enc_a = MD_E_attr(
                opts.input_dim, output_nc=self.nz, c_dim=opts.num_domains)
            self.gen = MD_G_multi(
                opts.input_dim, nz=self.nz, c_dim=opts.num_domains)

        self.dis1_opt = torch.optim.Adam(
            self.dis1.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.dis2_opt = torch.optim.Adam(
            self.dis2.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_c_opt = torch.optim.Adam(
            self.enc_c.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.enc_a_opt = torch.optim.Adam(
            self.enc_a.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.gen_opt = torch.optim.Adam(
            self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.disContent = MD_Dis_content(c_dim=opts.num_domains)
        self.disContent_opt = torch.optim.Adam(self.disContent.parameters(
        ), lr=lr/2.5, betas=(0.5, 0.999), weight_decay=0.0001)
        self.cls_loss = nn.BCEWithLogitsLoss()

    def initialize(self):
        self.dis1.apply(gaussian_weights_init)
        self.dis2.apply(gaussian_weights_init)
        self.disContent.apply(gaussian_weights_init)
        self.gen.apply(gaussian_weights_init)
        self.enc_c.apply(gaussian_weights_init)
        self.enc_a.apply(gaussian_weights_init)

    def set_scheduler(self, opts, last_ep=0):
        self.dis1_sch = get_scheduler(self.dis1_opt, opts, last_ep)
        self.dis2_sch = get_scheduler(self.dis2_opt, opts, last_ep)
        print(self.dis2_opt)
        self.disContent_opt.param_groups[0]['initial_lr'] = 0.00004
        print(self.disContent_opt)
        self.disContent_sch = get_scheduler(
            self.disContent_opt, opts, last_ep)
        self.enc_c_sch = get_scheduler(self.enc_c_opt, opts, last_ep)
        self.enc_a_sch = get_scheduler(self.enc_a_opt, opts, last_ep)
        self.gen_sch = get_scheduler(self.gen_opt, opts, last_ep)

    def update_lr(self):
        self.dis1_sch.step()
        self.dis2_sch.step()
        self.disContent_sch.step()
        self.enc_c_sch.step()
        self.enc_a_sch.step()
        self.gen_sch.step()

    def setgpu(self, gpu):
        self.gpu = gpu
        self.dis1.cuda(self.gpu)
        self.dis2.cuda(self.gpu)
        self.enc_c.cuda(self.gpu)
        self.enc_a.cuda(self.gpu)
        self.gen.cuda(self.gpu)
        self.disContent.cuda(self.gpu)

    def get_z_random(self, batchSize, nz, random_type='gauss'):
        z = torch.randn(batchSize, nz)
        return z

    def test_forward_random(self, image):
        self.z_content = self.enc_c.forward(image)
        outputs = []
        for i in range(self.opts.num_domains):
            self.z_random = self.get_z_random(image.size(0), self.nz, 'gauss')
            c_trg = np.zeros((image.size(0), self.opts.num_domains))
            c_trg[:, i] = 1
            c_trg = torch.FloatTensor(c_trg)
            output = self.gen.forward(self.z_content, self.z_random, c_trg)
            outputs.append(output)
        return outputs

    def test_forward_transfer(self, image, image_trg, c_trg):
        self.z_content = self.enc_c.forward(image)
        self.mu, self.logvar = self.enc_a.forward(self.image_trg, self.c_trg)
        std = torch.exp(self.logvar.mul(0.5))
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
        self.z_attr = torch.add(eps.mul(std), self.mu)
        output = self.gen.forward(self.z_content, self.z_attr, c_trg)
        return output

    def forward(self):
        # input images
        if not self.input.size(0) % 2 == 0:
            print("Need to be even QAQ")
            input()
        half_size = self.input.size(0)//2
        self.real_A = self.input[0:half_size]
        self.real_B = self.input[half_size:]
        c_org_A = self.c_org[0:half_size]
        c_org_B = self.c_org[half_size:]

        # get encoded z_c
        self.real_img = torch.cat((self.real_A, self.real_B), 0)
        self.z_content = self.enc_c.forward(self.real_img)
        self.z_content_a, self.z_content_b = torch.split(
            self.z_content, half_size, dim=0)

        # get encoded z_a
        if self.concat:
            self.mu, self.logvar = self.enc_a.forward(
                self.real_img, self.c_org)
#             print('output of enc_a')
#             print(self.mu)
#             print(self.logvar)
            # <-------- inf values are obtained in this operations
            std = torch.exp(self.logvar.mul(0.5))
            eps = self.get_z_random(
                std.size(0), std.size(1), 'gauss').to(device)
            self.z_attr = torch.add(eps.mul(std), self.mu)
#             print('self.z_attr')
#             print(self.z_attr)
        else:
            self.z_attr = self.enc_a.forward(self.real_img, self.c_org)
        self.z_attr_a, self.z_attr_b = torch.split(
            self.z_attr, half_size, dim=0)
        # get random z_a
        self.z_random = self.get_z_random(
            half_size, self.nz, 'gauss').to(device)

        # first cross translation
        input_content_forA = torch.cat(
            (self.z_content_b, self.z_content_a, self.z_content_b), 0)
        input_content_forB = torch.cat(
            (self.z_content_a, self.z_content_b, self.z_content_a), 0)
        input_attr_forA = torch.cat(
            (self.z_attr_a, self.z_attr_a, self.z_random), 0)
        input_attr_forB = torch.cat(
            (self.z_attr_b, self.z_attr_b, self.z_random), 0)
#         print('self.z_attr_a, self.z_attr_a, self.z_random')
#         print(self.z_attr_a)
#         print(self.z_attr_a)
#         print(self.z_random)
        input_c_forA = torch.cat((c_org_A, c_org_A, c_org_A), 0)
        input_c_forB = torch.cat((c_org_B, c_org_B, c_org_B), 0)
#         print('gen fakeA')
        output_fakeA = self.gen.forward(
            input_content_forA, input_attr_forA, input_c_forA)
#         print('--nan values generated ---------------------')
#         print('gen fakeB')
        output_fakeB = self.gen.forward(
            input_content_forB, input_attr_forB, input_c_forB)
#         print('--nan values generated ---------------------')
        self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random = torch.split(
            output_fakeA, self.z_content_a.size(0), dim=0)
        self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(
            output_fakeB, self.z_content_a.size(0), dim=0)

        # get reconstructed encoded z_c
        self.fake_encoded_img = torch.cat(
            (self.fake_A_encoded, self.fake_B_encoded), 0)
        self.z_content_recon = self.enc_c.forward(self.fake_encoded_img)
        self.z_content_recon_b, self.z_content_recon_a = torch.split(
            self.z_content_recon, half_size, dim=0)

        # get reconstructed encoded z_a
        if self.concat:
            self.mu_recon, self.logvar_recon = self.enc_a.forward(
                self.fake_encoded_img, self.c_org)
            std_recon = torch.exp(self.logvar_recon.mul(0.5))
            eps_recon = self.get_z_random(std_recon.size(
                0), std_recon.size(1), 'gauss').to(device)
            self.z_attr_recon = torch.add(
                eps_recon.mul(std_recon), self.mu_recon)
        else:
            self.z_attr_recon = self.enc_a.forward(
                self.fake_encoded_img, self.c_org)
        self.z_attr_recon_a, self.z_attr_recon_b = torch.split(
            self.z_attr_recon, half_size, dim=0)

        # second cross translation
#         print('gen fakeA_recon')
        self.fake_A_recon = self.gen.forward(
            self.z_content_recon_a, self.z_attr_recon_a, c_org_A)
#         print('gen fakeB_recon')
        self.fake_B_recon = self.gen.forward(
            self.z_content_recon_b, self.z_attr_recon_b, c_org_B)

        # for display
        self.image_display = torch.cat((self.real_A[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(),
                                        self.fake_B_random[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach(
        ).cpu(), self.fake_A_recon[0:1].detach().cpu(),
            self.real_B[0:1].detach().cpu(
        ), self.fake_A_encoded[0:1].detach().cpu(),
            self.fake_A_random[0:1].detach().cpu(), self.fake_BB_encoded[0:1].detach().cpu(), self.fake_B_recon[0:1].detach().cpu()), dim=0)

        # for latent regression
        self.fake_random_img = torch.cat(
            (self.fake_A_random, self.fake_B_random), 0)
        if self.concat:
            self.mu2, _ = self.enc_a.forward(self.fake_random_img, self.c_org)
            self.mu2_a, self.mu2_b = torch.split(self.mu2, half_size, 0)
        else:
            self.z_attr_random = self.enc_a.forward(
                self.fake_random_img, self.c_org)
            self.z_attr_random_a, self.z_attr_random_b = torch.split(
                self.z_attr_random, half_size, 0)

#         print('forward done')

    def update_D_content(self, image, c_org):
        self.input = image
        self.z_content = self.enc_c.forward(self.input)
        self.disContent_opt.zero_grad()
        pred_cls = self.disContent.forward(self.z_content.detach())
        loss_D_content = self.cls_loss(pred_cls, c_org)
        loss_D_content.backward()
        self.disContent_loss = loss_D_content.item()
        nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
        self.disContent_opt.step()

    def update_D(self, image, c_org):
        self.input = image
        self.c_org = c_org
        self.forward()

        self.dis1_opt.zero_grad()
        self.D1_gan_loss, self.D1_cls_loss = self.backward_D(
            self.dis1, self.input, self.fake_encoded_img)
        self.dis1_opt.step()

        self.dis2_opt.zero_grad()
        self.D2_gan_loss, self.D2_cls_loss = self.backward_D(
            self.dis2, self.input, self.fake_random_img)
        self.dis2_opt.step()

    def backward_D(self, netD, real, fake):
        pred_fake, pred_fake_cls = netD.forward(fake.detach())
        pred_real, pred_real_cls = netD.forward(real)

        loss_D_gan = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all0 = torch.zeros_like(out_fake)
            all1 = torch.ones_like(out_real)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            loss_D_gan += ad_true_loss + ad_fake_loss

        loss_D_cls = self.cls_loss(pred_real_cls, self.c_org)
        loss_D = loss_D_gan + self.opts.lambda_cls * loss_D_cls
        loss_D.backward()
        return loss_D_gan, loss_D_cls

    def update_EG(self):
        # update G, Ec, Ea
        self.enc_c_opt.zero_grad()
        self.enc_a_opt.zero_grad()
        self.gen_opt.zero_grad()
        self.backward_EG()
        self.backward_G_alone()
        self.enc_c_opt.step()
        self.enc_a_opt.step()
        self.gen_opt.step()

    def backward_EG(self):
        # content Ladv for generator
        loss_G_GAN_content = self.backward_G_GAN_content(self.z_content)

        # Ladv for generator
        pred_fake, pred_fake_cls = self.dis1.forward(self.fake_encoded_img)
        loss_G_GAN = 0
        for out_a in pred_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake)
            loss_G_GAN += nn.functional.binary_cross_entropy(
                outputs_fake, all_ones)

        # classification
        loss_G_cls = self.cls_loss(
            pred_fake_cls, self.c_org) * self.opts.lambda_cls_G

        # self and cross-cycle recon
        loss_G_L1_self = torch.mean(torch.abs(
            self.input - torch.cat((self.fake_AA_encoded, self.fake_BB_encoded), 0))) * self.opts.lambda_rec
        loss_G_L1_cc = torch.mean(torch.abs(
            self.input - torch.cat((self.fake_A_recon, self.fake_B_recon), 0))) * self.opts.lambda_rec

        # KL loss - z_c
        loss_kl_zc = self._l2_regularize(self.z_content) * 0.01

        # KL loss - z_a
        if self.concat:
            kl_element = torch.add(
                (- torch.add(self.mu.pow(2), self.logvar.exp())) + 1, self.logvar)
            loss_kl_za = torch.sum(kl_element) * (-0.5) * 0.01
        else:
            loss_kl_za = self._l2_regularize(self.z_attr) * 0.01

        loss_G = loss_G_GAN + loss_G_cls + loss_G_L1_self + \
            loss_G_L1_cc + loss_kl_zc + loss_kl_za
        loss_G += loss_G_GAN_content
        loss_G.backward(retain_graph=True)

        self.gan_loss = loss_G_GAN.item()
        self.gan_cls_loss = loss_G_cls.item()
        self.gan_loss_content = loss_G_GAN_content.item()
        self.kl_loss_zc = loss_kl_zc.item()
        self.kl_loss_za = loss_kl_za.item()
        self.l1_self_rec_loss = loss_G_L1_self.item()
        self.l1_cc_rec_loss = loss_G_L1_cc.item()
        self.G_loss = loss_G.item()

    def backward_G_GAN_content(self, data):
        pred_cls = self.disContent.forward(data)
        loss_G_content = self.cls_loss(pred_cls, 1-self.c_org)
        return loss_G_content

    def backward_G_alone(self):
        # Ladv for generator
        pred_fake, pred_fake_cls = self.dis2.forward(self.fake_random_img)
        loss_G_GAN2 = 0
        for out_a in pred_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake)
            loss_G_GAN2 += nn.functional.binary_cross_entropy(
                outputs_fake, all_ones)

        # classification
        loss_G_cls2 = self.cls_loss(
            pred_fake_cls, self.c_org) * self.opts.lambda_cls_G

        # latent regression loss
        if self.concat:
            loss_z_L1_a = torch.mean(
                torch.abs(self.mu2_a - self.z_random)) * 10
            loss_z_L1_b = torch.mean(
                torch.abs(self.mu2_b - self.z_random)) * 10
        else:
            loss_z_L1_a = torch.mean(
                torch.abs(self.z_attr_random_a - self.z_random)) * 10
            loss_z_L1_b = torch.mean(
                torch.abs(self.z_attr_random_b - self.z_random)) * 10

        loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_G_GAN2 + loss_G_cls2
        loss_z_L1.backward()
        self.l1_recon_z_loss = loss_z_L1_a.item() + loss_z_L1_b.item()
        self.gan2_loss = loss_G_GAN2.item()
        self.gan2_cls_loss = loss_G_cls2.item()

    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def update_lr(self):
        self.dis1_sch.step()
        self.dis2_sch.step()
        self.enc_c_sch.step()
        self.enc_a_sch.step()
        self.gen_sch.step()

    def assemble_outputs(self):
        images_a = self.normalize_image(self.real_A).detach()
        images_b = self.normalize_image(self.real_B).detach()
        images_a1 = self.normalize_image(self.fake_A_encoded).detach()
        images_a2 = self.normalize_image(self.fake_A_random).detach()
        images_a3 = self.normalize_image(self.fake_A_recon).detach()
        images_a4 = self.normalize_image(self.fake_AA_encoded).detach()
        images_b1 = self.normalize_image(self.fake_B_encoded).detach()
        images_b2 = self.normalize_image(self.fake_B_random).detach()
        images_b3 = self.normalize_image(self.fake_B_recon).detach()
        images_b4 = self.normalize_image(self.fake_BB_encoded).detach()
        row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::],
                         images_b2[0:1, ::], images_a4[0:1, ::], images_a3[0:1, ::]), 3)
        row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::],
                         images_a2[0:1, ::], images_b4[0:1, ::], images_b3[0:1, ::]), 3)
        return torch.cat((row1, row2), 2)

    def normalize_image(self, x):
        return x[:, 0:3, :, :]

    def save(self, filename, ep, total_it):
        state = {
            'dis1': self.dis1.state_dict(),
            'dis2': self.dis2.state_dict(),
            'disContent': self.disContent.state_dict(),
            'enc_c': self.enc_c.state_dict(),
            'enc_a': self.enc_a.state_dict(),
            'gen': self.gen.state_dict(),
            'dis1_opt': self.dis1_opt.state_dict(),
            'dis2_opt': self.dis2_opt.state_dict(),
            'disContent_opt': self.disContent_opt.state_dict(),
            'enc_c_opt': self.enc_c_opt.state_dict(),
            'enc_a_opt': self.enc_a_opt.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        return

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
        # weight
        if train:
            self.dis1.load_state_dict(checkpoint['dis1'])
            self.dis2.load_state_dict(checkpoint['dis2'])
            self.disContent.load_state_dict(checkpoint['disContent'])
        self.enc_c.load_state_dict(checkpoint['enc_c'])
        self.enc_a.load_state_dict(checkpoint['enc_a'])
        self.gen.load_state_dict(checkpoint['gen'])
        # optimizer
        if train:
            self.dis1_opt.load_state_dict(checkpoint['dis1_opt'])
            self.dis2_opt.load_state_dict(checkpoint['dis2_opt'])
            self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
            self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
            self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        return checkpoint['ep'], checkpoint['total_it']

class HistAuGANNormalizer(BaseNormalizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.opts = Args()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.domain = np.random.randint(5)
        self.model = MD_multi(self.opts)
        self.model.resume(self.opts.resume, train=False)
        self.model.to(self.device)
        self.model.eval()

    def norm(self, image):
        return image.divide(255).sub(0.5).mul(2)

    def un_norm(self, image):
        return image.mul(0.5).add(0.5).mul(255).type(torch.uint8)
    
    def fit(self, target_image):
        print("HistAuGANNormalizer does not need to be fitted -> pass")
        pass

    def transform(self, image):
        image = pil_to_tensor(image).to(self.device)
        z_content = self.model.enc_c(self.norm(image).unsqueeze(0))
        out = generate_hist_augs(image, self.domain, self.model, z_content, new_domain=3, stats=(mean_domains, std_domains), device=self.device)
        out = to_pil_image(self.un_norm(out))
        return out




