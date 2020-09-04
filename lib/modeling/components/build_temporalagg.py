import torch
import torch.nn as nn

# Take average of K frames
class TemporalAggregation_Mean(nn.Module):
    def __init__(self, cfg):
        super(TemporalAggregation_Mean, self).__init__()
        self.K = cfg.K

    def forward(self, s):
        s = s.view(s.size(0) // self.K, self.K, s.size(1), s.size(2), s.size(3))
        s = torch.mean(s, dim=1)

        return s

# Concat K frames together
class TemporalAggregation_Cat(nn.Module):
    def __init__(self, cfg):
        super(TemporalAggregation_Cat, self).__init__()
        self.K = cfg.K

    def forward(self, s):
        s = s.view(s.size(0) // self.K, self.K*s.size(1), s.size(2), s.size(3))

        return s

# Do simple 2D Conv on each frame separately
class TemporalAggregation_ConvCat(nn.Module):
    def __init__(self, cfg, in_channels):
        super(TemporalAggregation_ConvCat, self).__init__()
        self.K = cfg.K

        self.conv = [nn.Conv2d(in_channels // self.K, in_channels // self.K, kernel_size=3, padding=1) for i in range (self.K)]

        self.conv = nn.ModuleList(self.conv)

    def forward(self, s):
        y = s.view(int(s.size(0) / self.K), s.size(1) * self.K, s.size(2), s.size(3))

        per_frame_feat = list()
        for i in range(self.K):
            f_size = s.size(1)
            f_i = y[:, i * f_size:(i + 1) * f_size, :, :]  # [B, C, H, W]
            per_frame_feat.append(self.conv[i](f_i))

        return torch.cat(per_frame_feat, 1)