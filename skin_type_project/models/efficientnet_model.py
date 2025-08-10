# import torch.nn as nn
# from torchvision.models import efficientnet_b0

# def get_model(num_classes=3, dropout_rate=0.4):
#     model = efficientnet_b0(pretrained=True)
    
#     # Thay thế classifier bằng một chuỗi Sequential gồm Dropout + Linear
#     in_features = model.classifier[1].in_features
#     model.classifier = nn.Sequential(
#         nn.Dropout(p=dropout_rate),
#         nn.Linear(in_features, num_classes)
#     )
    
#     return model

import torch
import torch.nn as nn

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x / keep_prob * binary_tensor

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        hidden_dim = int(in_channels * se_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride, drop_path=0.0, se_ratio=0.25):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expansion

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if expansion != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        self.se = SqueezeExcite(hidden_dim, se_ratio)

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = self.drop_path(x)
            x = x + residual
        return x

class EfficientNetB0Lite(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5) :
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        # ➤ Rút gọn mỗi nhóm chỉ còn 1 block
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, expansion=1, stride=1, drop_path=0.01),
            MBConvBlock(16, 24, expansion=6, stride=2, drop_path=0.01),
            MBConvBlock(24, 40, expansion=6, stride=2, drop_path=0.02),
            MBConvBlock(40, 80, expansion=6, stride=2, drop_path=0.03),
            MBConvBlock(80, 112, expansion=6, stride=1, drop_path=0.04),
            MBConvBlock(112, 192, expansion=6, stride=2, drop_path=0.05),
            MBConvBlock(192, 320, expansion=6, stride=1, drop_path=0.05),
        )

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(1280, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def get_model(num_classes=3):
    return EfficientNetB0Lite(num_classes)

