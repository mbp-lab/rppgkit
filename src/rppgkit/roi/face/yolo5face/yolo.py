# SPDX-License-Identifier: GPL-3.0-only
# Copied third-party code; origin and license in THIRD_PARTY_NOTICES.md.
"""YOLO5Face network: verbatim copies of the original code.

Every function and class below is copied unchanged from YOLO5Face (Qi et al.,
https://github.com/deepcam-cn/yolov5-face). Only detection-related code is included.
Deviations from the originals, each marked "rppgkit:" in the code:
  - imports point to this package
  - Model: unused methods _print_biases, fuse, nms, autoshape removed
  - parse_model: its module-type list trimmed to the modules this architecture uses
Sources: model/yolo.py (Detect, Model, parse_model), utils/common.py (autopad, Conv,
StemBlock, Bottleneck, C3, SPP, Concat), utils/torch_utils.py (time_synchronized,
initialize_weights, model_info, scale_img).
"""

import logging
import math
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rppgkit.roi.face.yolo5face.data_ops import check_anchor_order, make_divisible

logger = logging.getLogger(__name__)

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            nn.SiLU()
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )
        # self.act = self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class StemBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, act=True):
        super(StemBlock, self).__init__()
        self.stem_1 = Conv(c1, c2, k, s, p, g, act)
        self.stem_2a = Conv(c2, c2 // 2, 1, 1, 0)
        self.stem_2b = Conv(c2 // 2, c2, 3, 2, 1)
        self.stem_2p = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.stem_3 = Conv(c2 * 2, c2, 1, 1, 0)

    def forward(self, x):
        stem_1_out = self.stem_1(x)
        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)
        stem_2p_out = self.stem_2p(stem_1_out)
        out = self.stem_3(torch.cat((stem_2b_out, stem_2p_out), 1))
        return out


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self, c1, c2, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        stride = int(model.stride.max()) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
    except (ImportError, Exception):
        fs = ''

    logger.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


_DEFAULT_CFG = str(Path(__file__).parent / "yolo5face.yaml")


class Detect(nn.Module):
    stride = None  # strides computed during build
    export_cat = False  # onnx export cat output

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        # self.no = nc + 5  # number of outputs per anchor
        self.no = nc + 5 + 10  # number of outputs per anchor

        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        if self.export_cat:
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
                x[i] = (
                    x[i]
                    .view(bs, self.na, self.no, ny, nx)
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                )

                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    self.grid[i], self.anchor_grid[i] = self._make_grid_new(nx, ny, i)

                y = torch.full_like(x[i], 0)
                y = y + torch.cat(
                    (
                        x[i][:, :, :, :, 0:5].sigmoid(),
                        torch.cat(
                            (
                                x[i][:, :, :, :, 5:15],
                                x[i][:, :, :, :, 15 : 15 + self.nc].sigmoid(),
                            ),
                            4,
                        ),
                    ),
                    4,
                )

                box_xy = (
                    y[:, :, :, :, 0:2] * 2.0 - 0.5 + self.grid[i].to(x[i].device)
                ) * self.stride[i]  # xy
                box_wh = (y[:, :, :, :, 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                # box_conf = torch.cat((box_xy, torch.cat((box_wh, y[:, :, :, :, 4:5]), 4)), 4)

                landm1 = (
                    y[:, :, :, :, 5:7] * self.anchor_grid[i]
                    + self.grid[i].to(x[i].device) * self.stride[i]
                )  # landmark x1 y1
                landm2 = (
                    y[:, :, :, :, 7:9] * self.anchor_grid[i]
                    + self.grid[i].to(x[i].device) * self.stride[i]
                )  # landmark x2 y2
                landm3 = (
                    y[:, :, :, :, 9:11] * self.anchor_grid[i]
                    + self.grid[i].to(x[i].device) * self.stride[i]
                )  # landmark x3 y3
                landm4 = (
                    y[:, :, :, :, 11:13] * self.anchor_grid[i]
                    + self.grid[i].to(x[i].device) * self.stride[i]
                )  # landmark x4 y4
                landm5 = (
                    y[:, :, :, :, 13:15] * self.anchor_grid[i]
                    + self.grid[i].to(x[i].device) * self.stride[i]
                )  # landmark x5 y5
                # landm = torch.cat((landm1, torch.cat((landm2, torch.cat((landm3, torch.cat((landm4, landm5), 4)), 4)), 4)), 4)
                # y = torch.cat((box_conf, torch.cat((landm, y[:, :, :, :, 15:15+self.nc]), 4)), 4)
                y = torch.cat(
                    [
                        box_xy,
                        box_wh,
                        y[:, :, :, :, 4:5],
                        landm1,
                        landm2,
                        landm3,
                        landm4,
                        landm5,
                        y[:, :, :, :, 15 : 15 + self.nc],
                    ],
                    -1,
                )

                z.append(y.view(bs, -1, self.no))
            return torch.cat(z, 1)

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = torch.full_like(x[i], 0)
                class_range = list(range(5)) + list(range(15, 15 + self.nc))
                y[..., class_range] = x[i][..., class_range].sigmoid()
                y[..., 5:15] = x[i][..., 5:15]
                # y = x[i].sigmoid()

                y[..., 0:2] = (
                    y[..., 0:2] * 2.0 - 0.5 + self.grid[i].to(x[i].device)
                ) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                # y[..., 5:15] = y[..., 5:15] * 8 - 4
                y[..., 5:7] = (
                    y[..., 5:7] * self.anchor_grid[i]
                    + self.grid[i].to(x[i].device) * self.stride[i]
                )  # landmark x1 y1
                y[..., 7:9] = (
                    y[..., 7:9] * self.anchor_grid[i]
                    + self.grid[i].to(x[i].device) * self.stride[i]
                )  # landmark x2 y2
                y[..., 9:11] = (
                    y[..., 9:11] * self.anchor_grid[i]
                    + self.grid[i].to(x[i].device) * self.stride[i]
                )  # landmark x3 y3
                y[..., 11:13] = (
                    y[..., 11:13] * self.anchor_grid[i]
                    + self.grid[i].to(x[i].device) * self.stride[i]
                )  # landmark x4 y4
                y[..., 13:15] = (
                    y[..., 13:15] * self.anchor_grid[i]
                    + self.grid[i].to(x[i].device) * self.stride[i]
                )  # landmark x5 y5

                # y[..., 5:7] = (y[..., 5:7] * 2 -1) * self.anchor_grid[i]  # landmark x1 y1
                # y[..., 7:9] = (y[..., 7:9] * 2 -1) * self.anchor_grid[i]  # landmark x2 y2
                # y[..., 9:11] = (y[..., 9:11] * 2 -1) * self.anchor_grid[i]  # landmark x3 y3
                # y[..., 11:13] = (y[..., 11:13] * 2 -1) * self.anchor_grid[i]  # landmark x4 y4
                # y[..., 13:15] = (y[..., 13:15] * 2 -1) * self.anchor_grid[i]  # landmark x5 y5

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def _make_grid_new(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if (
            "1.10.0" in torch.__version__
        ):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(
                [torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing="ij"
            )
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (
            (self.anchors[i].clone() * self.stride[i])
            .view((1, self.na, 1, 1, 2))
            .expand((1, self.na, ny, nx, 2))
            .float()
        )
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(
        self,
        cfg=_DEFAULT_CFG,
        ch=3,
        nc=None,
    ):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            logger.info(
                "Overriding model.yaml nc=%g with nc=%g" % (self.yaml["nc"], nc)
            )
            self.yaml["nc"] = nc  # override yaml value
        self.model, self.save = parse_model(
            deepcopy(self.yaml), ch=[ch]
        )  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]
            )  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info("")

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers

            if profile:
                o = (
                    thop.profile(m, inputs=(x,), verbose=False)[0] / 1e9 * 2
                    if thop
                    else 0
                )  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print("%10.1f%10.0f%10.1fms %-40s" % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print("%.1fms total" % sum(dt))
        return x

    def _initialize_biases(
        self, cf=None
    ):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(
                8 / (640 / s) ** 2
            )  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99))
                if cf is None
                else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights




    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info(
        "\n%3s%18s%3s%10s  %-40s%-30s"
        % ("", "from", "n", "params", "module", "arguments")
    )
    anchors, nc, gd, gw = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
    )
    na = (
        (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    )  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(
        d["backbone"] + d["head"]
    ):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # rppgkit: list trimmed to the modules copied here (original also listed DWConv,
        # MixConv2d, Focus, CrossConv, BottleneckCSP, ShuffleV2Block, BlazeBlock,
        # DoubleBlazeBlock, which yolo5face.yaml does not use).
        if m in [
            Conv,
            Bottleneck,
            SPP,
            C3,
            StemBlock,
        ]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [C3]:  # rppgkit: BottleneckCSP removed (not copied, not used)
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = (
            nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        )  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = (
            i,
            f,
            t,
            np,
        )  # attach index, 'from' index, type, number params
        logger.info("%3s%18s%3s%10.0f  %-40s%-30s" % (i, f, n, np, t, args))  # print
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
