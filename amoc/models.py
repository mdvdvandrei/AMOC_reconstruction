import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.vision_transformer import VisionTransformer
import math




class ResidualCNNHetBaseline(nn.Module):
    """
    Residual CNN with μ/logσ² heads + flat skip to both.
    If model_args.flat_hw provided -> fixed Linear; else LazyLinear.
    """
    def __init__(self, in_channels: int, out_dim: int, flat_hw=None, p_drop: float = 0.3, tanh_scale: float = 8.0):
        super().__init__()
        self.out_dim = out_dim
        self.tanh_scale = tanh_scale

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.layer1 = ResidualBlock_new(16,  32,  2, p_drop)
        self.layer2 = ResidualBlock_new(32,  64,  2, p_drop)
        self.layer3 = ResidualBlock_new(64,  128, 2, p_drop)
        self.layer4 = ResidualBlock_new(128, 256, 2, p_drop)
        self.pool   = nn.AdaptiveAvgPool2d(1)

        self.mu_head = nn.Sequential(
            nn.Linear(256, 64), nn.LeakyReLU(0.01, inplace=True), nn.Dropout(0.5),
            nn.Linear(64, out_dim)
        )
        self.log_head = nn.Sequential(
            nn.Linear(256, 64), nn.LeakyReLU(0.01, inplace=True), nn.Dropout(0.5),
            nn.Linear(64, out_dim)
        )

        if flat_hw:
            H, W = flat_hw
            flat_dim = in_channels * H * W
            self.skip_mu  = nn.Linear(flat_dim, out_dim)
            self.skip_log = nn.Linear(flat_dim, out_dim)
        else:
            self.skip_mu  = nn.LazyLinear(out_dim)
            self.skip_log = nn.LazyLinear(out_dim)

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        b = x.size(0)
        flat = x.view(b, -1)
        h = self.stem(x)
        h = self.layer4(self.layer3(self.layer2(self.layer1(h))))
        trunk = self.pool(h).view(b, -1)
        mu     = self.mu_head(trunk)  + self.skip_mu(flat)
        logvar = self.log_head(trunk) + self.skip_log(flat)
        s = self.tanh_scale
        logvar = s * torch.tanh(logvar / s)
        return mu, logvar
        

class ChannelLayerNorm(nn.Module):
    def __init__(self, C, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(C, eps=eps)
    def forward(self, x):
        return self.ln(x.permute(0,2,3,1)).permute(0,3,1,2)

class GaussianBlur2d(nn.Module):
    """Fixed 3×3 depthwise Gaussian blur as anti-alias before stride-4 stem."""
    def __init__(self, channels, sigma=1.0):
        super().__init__()
        k = 3
        grid = torch.arange(k) - (k//2)
        xx, yy = torch.meshgrid(grid, grid, indexing="ij")
        ker = torch.exp(-(xx**2 + yy**2)/(2*sigma**2))
        ker = (ker / ker.sum()).float()
        weight = ker.view(1,1,k,k).repeat(channels,1,1,1)
        conv = nn.Conv2d(channels, channels, k, stride=1, padding=1, groups=channels, bias=False)
        with torch.no_grad():
            conv.weight.copy_(weight)
        for p in conv.parameters(): p.requires_grad = False
        self.conv = conv
    def forward(self, x): return self.conv(x)

class CNXBlock(nn.Module):
    def __init__(self, dim, layer_scale_init=1e-6):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.ln = ChannelLayerNorm(dim)
        self.pw1 = nn.Conv2d(dim, 4*dim, kernel_size=1, bias=True)
        self.pw2 = nn.Conv2d(4*dim, dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.gamma = nn.Parameter(layer_scale_init * torch.ones((dim,1,1))) if layer_scale_init>0 else None
    def forward(self, x):
        r = x
        x = self.dw(x)
        x = self.ln(x)
        x = self.pw2(self.act(self.pw1(x)))
        if self.gamma is not None: x = self.gamma * x
        return r + x

class ConvNeXtTrunk(nn.Module):
    def __init__(self, in_ch=1, dims=(48,96,192,384), depths=(3,3,9,3),
                 anti_alias=True, blur_sigma=1.0):
        super().__init__()
        C1,C2,C3,C4 = dims; d1,d2,d3,d4 = depths
        self.aa = GaussianBlur2d(in_ch, sigma=blur_sigma) if anti_alias else None
        self.stem   = nn.Sequential(nn.Conv2d(in_ch, C1, 4, 4, bias=True), ChannelLayerNorm(C1))  # 144x108→36x27
        self.stage1 = nn.Sequential(*[CNXBlock(C1) for _ in range(d1)])
        self.down12 = nn.Sequential(ChannelLayerNorm(C1), nn.Conv2d(C1, C2, 2, 2, bias=True))     # →18x13
        self.stage2 = nn.Sequential(*[CNXBlock(C2) for _ in range(d2)])
        self.down23 = nn.Sequential(ChannelLayerNorm(C2), nn.Conv2d(C2, C3, 2, 2, bias=True))     # →9x6
        self.stage3 = nn.Sequential(*[CNXBlock(C3) for _ in range(d3)])
        self.down34 = nn.Sequential(ChannelLayerNorm(C3), nn.Conv2d(C3, C4, 2, 2, bias=True))     # →4x3
        self.stage4 = nn.Sequential(*[CNXBlock(C4) for _ in range(d4)])
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.out_ch = C4
    def forward(self, x):
        if self.aa is not None: x = self.aa(x)
        x = self.stem(x)
        x = self.stage1(x); x = self.down12(x)
        x = self.stage2(x); x = self.down23(x)
        x = self.stage3(x); x = self.down34(x)
        x = self.stage4(x)
        return self.pool(x).flatten(1)

# ---- μ-only head (Stage-1) ----
class ConvNeXtRegressorMu(nn.Module):
    def __init__(self, in_ch, out_dim=1, flat_hw=(144,108),
                 dims=(48,96,192,384), depths=(3,3,9,3),
                 p_drop=0.3, use_flat_skip=False,
                 anti_alias=True, blur_sigma=1.0):
        super().__init__()
        self.trunk = ConvNeXtTrunk(in_ch=in_ch, dims=dims, depths=depths,
                                   anti_alias=anti_alias, blur_sigma=blur_sigma)
        C = self.trunk.out_ch
        self.mu_head = nn.Sequential(nn.Linear(C, 64), nn.GELU(), nn.Dropout(p_drop), nn.Linear(64, out_dim))
        self.use_flat_skip = bool(use_flat_skip)
        if self.use_flat_skip:
            H,W = flat_hw
            self.skip_mu = nn.Linear(in_ch*H*W, out_dim)
    def forward(self, x):
        z = self.trunk(x)
        mu = self.mu_head(z)
        if self.use_flat_skip:
            mu = mu + self.skip_mu(x.view(x.size(0), -1))
        return mu

# ---- heteroscedastic head (Stage-2 joint; we only use logvar) ----
class ConvNeXtHet(nn.Module):
    def __init__(self, in_ch=1, out_dim=1, flat_hw=(144,108),
                 dims=(48,96,192,384), depths=(3,3,9,3),
                 p_drop=0.3, tanh_scale=8.0, use_flat_skip=False,
                 anti_alias=True, blur_sigma=1.0):
        super().__init__()
        self.trunk = ConvNeXtTrunk(in_ch=in_ch, dims=dims, depths=depths,
                                   anti_alias=anti_alias, blur_sigma=blur_sigma)
        C = self.trunk.out_ch
        self.mu_head  = nn.Sequential(nn.Linear(C, 64), nn.GELU(), nn.Dropout(p_drop), nn.Linear(64, out_dim))
        self.log_head = nn.Sequential(nn.Linear(C, 64), nn.GELU(), nn.Dropout(p_drop), nn.Linear(64, out_dim))
        self.use_flat_skip = bool(use_flat_skip)
        if self.use_flat_skip:
            H,W = flat_hw
            self.skip_mu  = nn.Linear(in_ch*H*W, out_dim)
            self.skip_log = nn.Linear(in_ch*H*W, out_dim)
        self.tanh_scale = float(tanh_scale)
    def forward(self, x):
        z = self.trunk(x)
        mu     = self.mu_head(z)
        logvar = self.log_head(z)
        if self.use_flat_skip:
            flat = x.view(x.size(0), -1)
            mu     = mu     + self.skip_mu(flat)
            logvar = logvar + self.skip_log(flat)
        s = self.tanh_scale
        logvar = s * torch.tanh(logvar / s)
        return mu, logvar



class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(SimpleCNN, self).__init__()
        self.input_channels = input_channels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.drop1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.LeakyReLU(negative_slope=0.01)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1)
        self.relu4 = nn.LeakyReLU(negative_slope=0.01)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Calculate the number of features for the fully connected layer.
        self._calculate_fc_in_features()
        
        # Fully connected layers
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.fc_in_features, 128)
        self.relu5 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(128, output_dim)
        
        # Linear connection from input to output (flattening assumed input dims 144x108)
        self.input_linear = nn.Linear(input_channels * 144 * 108, output_dim)
    
    def _calculate_fc_in_features(self):
        dummy_input = torch.zeros(1, self.input_channels, 144, 108)
        x = self.conv1(dummy_input)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        
        self.fc_in_features = x.numel()
    
    def forward(self, x):
        input_flat = x.reshape(x.size(0), -1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        
        x = x.reshape(x.size(0), -1)
        x = self.drop2(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        input_connection = self.input_linear(input_flat)
        x = x + input_connection
        return x
    



class SimpleViT(nn.Module):
    def __init__(self, input_channels, output_dim, patch_size=8, embedding_dim=128, num_heads=8, num_layers=6, ff_dim=1024):
        super(SimpleViT, self).__init__()

        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim

        # Vision Transformer configuration
        self.vit = VisionTransformer(
            img_size=(144, 108),
            patch_size=patch_size,
            in_chans=input_channels,
            num_classes=output_dim,
            embed_dim=embedding_dim,
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.3,
            attn_drop_rate=0.3,
            drop_path_rate=0.3,
            norm_layer=nn.LayerNorm,
        )

    def forward(self, x):
        return self.vit(x)
'''
# Example usage
model = SimpleViT(input_channels=1, output_dim=1)  # 1-channel input for grayscale images
input_tensor = torch.randn(32, 1, 144, 108)  # Batch size of 32
output = model(input_tensor)
print(output.shape)  # Should be [32, 1]
print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
'''








# ----------------------------------------------------------------------
#  1.  Little CNN that converts (C, H, W) → d_model
# ----------------------------------------------------------------------
class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),      # → (B, 128, 1, 1)
            nn.Flatten(),                 # → (B, 128)
            nn.Linear(128, d_model),
        )

    def forward(self, x):                 # x: (B, C, H, W)
        return self.net(x)                # (B, d_model)


# ----------------------------------------------------------------------
#  2.  Positional encoding (sinusoidal – no extra params)
# ----------------------------------------------------------------------
def positional_encoding(max_len: int, d_model: int, device: torch.device):
    """Return (max_len, d_model) table."""
    pe = torch.zeros(max_len, d_model, device=device)
    pos = torch.arange(0, max_len, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device) *
                    (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ----------------------------------------------------------------------
#  3.  Complete model
# ----------------------------------------------------------------------
class TemporalTransformer(nn.Module):
    """
    (B, T, C, H, W)  →  predict y (vector or scalar)
    """
    def __init__(
        self,
        in_channels: int,
        output_dim: int = 1,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 512,
        max_seq_len: int = 120,      # plenty for monthly data
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=False,       # (T, B, d)
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.register_buffer(
            "pos_embed",
            positional_encoding(max_seq_len, d_model, device="cpu"),
            persistent=False,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, output_dim),
        )

    def forward(self, x):                    # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)           # merge batch & time
        feats = self.encoder(x)              # (B*T, d)
        feats = feats.view(B, T, -1)         # (B, T, d)
        feats = feats + self.pos_embed[:T]   # add PE  (broadcast on B)

        feats = feats.permute(1, 0, 2)       # → (T, B, d) for transformer
        feats = self.temporal(feats)         # (T, B, d)
        last = feats[-1]                     # last timestep, shape (B, d)
        return self.head(last)               # (B, output_dim)





