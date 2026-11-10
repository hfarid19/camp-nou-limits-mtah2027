"""Per-frame touch spotter: 2D ResNet-18 trunk with temporal shift (TSM) + dilated temporal conv head.

Input  x: (B, T, 3, S, S) float, ball-centred crops (ImageNet-normalised).
Output   : (B, T, K) logits; K=1 (touch) or 1+extra heads.
E2E-Spot / T-DEED style: dense per-frame prediction, peaks decoded with temporal NMS.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalShift(nn.Module):
    """Shift 1/8 of channels forward and 1/8 backward in time (Lin et al., TSM)."""

    def __init__(self, block, T, fold_div=8):
        super().__init__()
        self.block, self.T, self.fold_div = block, T, fold_div

    def forward(self, x):
        nt, c, h, w = x.shape
        T = self.T
        x5 = x.view(nt // T, T, c, h, w)
        f = c // self.fold_div
        out = torch.zeros_like(x5)
        out[:, 1:, :f] = x5[:, :-1, :f]
        out[:, :-1, f:2 * f] = x5[:, 1:, f:2 * f]
        out[:, :, 2 * f:] = x5[:, :, 2 * f:]
        return self.block(out.view(nt, c, h, w))


class TouchNet(nn.Module):
    def __init__(self, T=32, k_out=1, pretrained=True, hidden=256):
        super().__init__()
        import timm
        self.T = T
        self.trunk = timm.create_model("resnet18", pretrained=pretrained, num_classes=0,
                                       global_pool="avg")
        # TSM on conv1 of every basic block
        for layer in [self.trunk.layer1, self.trunk.layer2, self.trunk.layer3, self.trunk.layer4]:
            for b in layer:
                b.conv1 = TemporalShift(b.conv1, T)
        d = self.trunk.num_features
        self.head = nn.Sequential(
            nn.Conv1d(d, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4), nn.GELU(),
            nn.Conv1d(hidden, k_out, 1))

    def set_T(self, T):
        self.T = T
        for m in self.modules():
            if isinstance(m, TemporalShift):
                m.T = T

    def forward(self, x):
        B, T = x.shape[:2]
        if T != self.T:
            self.set_T(T)
        f = self.trunk(x.flatten(0, 1))            # (B*T, d)
        f = f.view(B, T, -1).transpose(1, 2)       # (B, d, T)
        return self.head(f).transpose(1, 2)        # (B, T, K)
