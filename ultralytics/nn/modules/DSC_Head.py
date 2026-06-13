import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .block import DFL
from ultralytics.utils.tal import dist2bbox, make_anchors


class Conv_GN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()

        if p is None:
            p = ((k - 1) // 2) * d

        self.conv = nn.Conv2d(
            c1, c2, k, s, p, groups=g, dilation=d, bias=False
        )

        num_groups = min(16, c2)
        while c2 % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        self.gn = nn.GroupNorm(num_groups, c2)
        self.act = nn.SiLU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale


class MLP(nn.Module):
    def __init__(self, c1, c2, c3, num_layers):
        super().__init__()

        layers = []

        if num_layers == 1:
            layers.append(nn.Conv2d(c1, c3, 1))
        else:
            layers.append(nn.Conv2d(c1, c2, 1))
            layers.append(nn.SiLU())

            for _ in range(num_layers - 2):
                layers.append(nn.Conv2d(c2, c2, 1))
                layers.append(nn.SiLU())

            layers.append(nn.Conv2d(c2, c3, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max):
        super(LQE, self).__init__()

        self.k = k
        self.reg_max = reg_max

        self.reg_conf = MLP(
            4 * (k + 1),
            hidden_dim,
            1,
            num_layers
        )

        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, C, H, W = pred_corners.size()

        prob = F.softmax(
            pred_corners.reshape(B, self.reg_max, 4, H, W),
            dim=1
        )

        prob_topk, _ = prob.topk(self.k, dim=1)

        stat = torch.cat(
            [prob_topk, prob_topk.mean(dim=1, keepdim=True)],
            dim=1
        )

        quality_score = self.reg_conf(
            stat.reshape(B, -1, H, W)
        )

        return scores + quality_score


class Detect_DSC_Head(nn.Module):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()

        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        self.format = None

        self.conv = nn.ModuleList(
            nn.Sequential(
                Conv_GN(x, hidc, 1),
                Conv_GN(hidc, hidc, 3, g=hidc)
            ) for x in ch
        )

        midc = hidc // 2
        self.share_conv = nn.Sequential(
            Conv_GN(hidc, midc, 1),
            Conv_GN(midc, midc, 3, g=midc),
            Conv_GN(midc, hidc, 1)
        )

        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)

        cls_c = max(hidc // 2, 64)
        self.cls_proj = Conv_GN(hidc, cls_c, 1)

        self.cls_refine = Conv_GN(cls_c, cls_c, 3, g=cls_c)

        self.cv3 = nn.Conv2d(cls_c, self.nc, 1)

        self.level_cls_bias = nn.Parameter(
            torch.zeros(self.nl, self.nc, 1, 1)
        )

        self.scale = nn.ModuleList(
            Scale(1.0) for _ in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.lqe = nn.ModuleList(
            LQE(4, 64, 2, self.reg_max) for _ in ch
        )

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            x[i] = self.share_conv(x[i])

            pred_corners = self.scale[i](
                self.cv2(x[i])
            )

            cls_feat = self.cls_proj(x[i])
            cls_feat = self.cls_refine(cls_feat)

            cls_logits = self.cv3(cls_feat) + self.level_cls_bias[i]

            pred_scores = self.lqe[i](
                cls_logits,
                pred_corners
            )

            x[i] = torch.cat(
                (pred_corners, pred_scores),
                dim=1
            )

        if self.training:
            return x

        shape = x[0].shape  # BCHW

        x_cat = torch.cat(
            [xi.view(shape[0], self.no, -1) for xi in x],
            dim=2
        )

        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1)
                for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        fmt = getattr(self, "format", None)

        if self.export and fmt in (
            "saved_model", "pb", "tflite", "edgetpu", "tfjs"
        ):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split(
                (self.reg_max * 4, self.nc),
                dim=1
            )

        dbox = self.decode_bboxes(box)

        if self.export and fmt in ("tflite", "edgetpu"):
            img_h = shape[2]
            img_w = shape[3]

            img_size = torch.tensor(
                [img_w, img_h, img_w, img_h],
                device=box.device
            ).reshape(1, 4, 1)

            norm = self.strides / (self.stride[0] * img_size)

            dbox = dist2bbox(
                self.dfl(box) * norm,
                self.anchors.unsqueeze(0) * norm[:, :2],
                xywh=True,
                dim=1
            )

        y = torch.cat(
            (dbox, cls.sigmoid()),
            dim=1
        )

        return y if self.export else (y, x)

    def bias_init(self):
        m = self

        m.cv2.bias.data[:] = 1.0

        m.cv3.bias.data[: m.nc] = math.log(
            5 / m.nc / (640 / 16) ** 2
        )

        if hasattr(m, "level_cls_bias"):
            nn.init.constant_(m.level_cls_bias, 0.0)

    def decode_bboxes(self, bboxes):
        return dist2bbox(
            self.dfl(bboxes),
            self.anchors.unsqueeze(0),
            xywh=True,
            dim=1
        ) * self.strides