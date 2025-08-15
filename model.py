import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryAwareAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, edge_weight=0.1):
        super(BoundaryAwareAttentionGate, self).__init__()
        
        self.edge_weight = edge_weight
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1)
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def compute_edge_map(self, x):
        x_gray = torch.mean(x, dim=1, keepdim=True)
        grad_x = F.conv2d(x_gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_gray, self.sobel_y, padding=1)
        edge = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        batch_size = edge.size(0)
        edge_flat = edge.view(batch_size, -1)
        edge_min = edge_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        edge_max = edge_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        edge = (edge - edge_min) / (edge_max - edge_min + 1e-6)
        
        return edge
        
    def forward(self, g, x):
        g_size = g.size()
        x_size = x.size()
        
        if g_size[2:] != x_size[2:]:
            g = F.interpolate(g, size=x_size[2:], mode='bilinear', align_corners=False)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi_raw = self.relu(g1 + x1)
        psi_raw = self.psi(psi_raw)
        
        edge = self.compute_edge_map(x)
        psi = torch.sigmoid(psi_raw + self.edge_weight * edge)
        
        return x * psi

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.in1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c + out_c, out_c)
        self.attention = BoundaryAwareAttentionGate(F_g=out_c, F_l=out_c, F_int=out_c//2, edge_weight=0.1)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        skip_att = self.attention(g=x, x=skip)
        x = torch.cat([x, skip_att], dim=1)
        x = self.conv(x)
        return x

class BoundaryAwareAttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = EncoderBlock(3, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        self.b = ConvBlock(512, 1024)
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        b = self.b(p4)
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        outputs = self.outputs(d4)
        return outputs