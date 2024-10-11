# modified from: https://github.com/lucidrains/vit-pytorch

import torch
from torch import nn, einsum
from einops import rearrange
from torch.autograd import Function

class StraightThrough(Function):
    @staticmethod
    def forward(ctx, input):
        index_1 = input[:, 0] > input[:, 1]
        ctx.save_for_backward(input, index_1)
        return index_1

    @staticmethod
    def backward(ctx, grad_output):
        input, index_1 = ctx.saved_tensors
        grad_input = grad_output.new_zeros(input.shape)
        grad_input[:, 0] = grad_output * index_1.float()
        grad_input[:, 1] = grad_output * (~index_1).float()
        return grad_input

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # shape = x.shape[:-1]
        # x = self.layers(x.view(-1, x.shape[-1]))
        x = self.layers(x)
        return x
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class WeightTransformer(nn.Module):
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        # self.pos_embedding = nn.Linear(2, dim)
        self.pos_embedding = MLP(2, dim, [64, dim])
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.out = nn.Softmax(dim=1)

    def forward(self, z, q):
        # z: local feature vector       [N*Q, 4, D]
        # q: local relative coordinate  [N*Q, 4, 2]
        b, n, _ = z.shape

        # Local position embedding
        z = z + self.pos_embedding(q)          # [N*Q, 4, D]
        z = self.dropout(z)

        # Attention module via Transformer
        # split z
        z_chunks = torch.chunk(z, n, dim=0)   
        z_transformed = [self.transformer(chunk) for chunk in z_chunks]  
        z = torch.cat(z_transformed, dim=0)  # [N*Q, 4, D]
        # z = self.transformer(z)             # [N*Q, 4, D]

        # Weight estimation
        z = self.mlp_head(z)                # [N*Q, 4, 1]
        z = self.out(z)

        return z

class DirectionWeightTransformer(nn.Module):
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        # self.pos_embedding = nn.Linear(2, dim)
        self.pos_embedding = MLP(2, 64, [32, 64])
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.direction_predictor = MLP(4*dim, 3, [4*dim, dim, dim])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.out = nn.Softmax(dim=1)

    def forward(self, z, q, c):
        # z: local feature vector       [N*Q, 4, D]
        # q: local relative coordinate  [N*Q, 4, 2]
        # c: coordinates                [N*Q, 4, 3]
        b, n, _ = z.shape

        # Local position embedding
        z = z + self.pos_embedding(q)          # [N*Q, 4, D]
        z = self.dropout(z)

        # direction prediction
        dir = self.direction_predictor(rearrange(z, 'n t d -> n (t d)')) # [N*Q, 3]
        dir = dir / torch.sqrt(torch.sum(dir**2))

        # Attention module via Transformer
        z = self.transformer(z)             # [N*Q, 4, D]

        # Weight estimation
        z = self.mlp_head(z)                # [N*Q, 4, 1]
        z = self.out(z)

        return z


class GateWeightTransformer(nn.Module):
    """
    通过线性层,将权重预测变成64维特征,并和原始特征相加
    """
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        # self.pos_embedding = nn.Linear(2, dim)
        self.pos_embedding = MLP(2, 64, [32, 64])
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer_1 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.transformer_2 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head_1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.mlp_head_2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.gatenet = nn.Sequential(
            MLP(4*dim, 2, [dim, dim]),
            nn.Softmax(dim=1)
        )

        self.connect = MLP(2, dim, [dim, dim]),
        
        self.out = nn.Softmax(dim=1)

    def forward(self, z, q):
        # z: local feature vector       [N*Q, 4, D]
        # q: local relative coordinate  [N*Q, 4, 2]
        b, n, _ = z.shape

        # Local position embedding
        z = z + self.pos_embedding(q)          # [N*Q, 4, D]
        z = self.dropout(z)

        # gate prediction
        z_agg = rearrange(z, 'n t d -> n (t d)') # [N*Q, 4*D]
        z_gate = self.gatenet(z_agg) # [N*Q, 2]

        # index for experts
        index_1 = z_gate[:,0]>z_gate[:,1] # [N*Q, 1]
        #index_1 = StraightThrough.apply(z_gate)
        index_2 = ~index_1 # [N*Q, 1]

        z_gate = self.connect(z_gate) # [N*Q, D]
        z_gate = torch.unsqueeze(z_gate, dim=1).repeat(1, 4, 1) # [N*Q,4, D]
        z = z + z_gate

        # select features
        z_1 = z[index_1] # [N1, 4, D]
        z_2 = z[index_2] # [N1, 4, D]

        # Attention module via Transformer
        z_1 = self.transformer_1(z_1)             # [N1, 4, D]
        z_2 = self.transformer_2(z_2)             # [N2, 4, D]
        
        # Weight estimation
        z_1 = self.mlp_head_1(z_1)                # [N1, 4, 1]
        z_2 = self.mlp_head_2(z_2)                # [N1, 4, 1]

        # concat the weight

        w = torch.empty([b, n, 1]).cuda()                # [N*Q, 4, 1]
        w[index_1] = z_1
        w[index_2] = z_2

        w = self.out(w)                           # [N*Q, 4, 1]

        return w, index_1, index_2

class GateWeightTransformer2(nn.Module):
    """
    通过线性层,将权重预测变成64维特征,并和原始特征相乘
    """
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        # self.pos_embedding = nn.Linear(2, dim)
        self.pos_embedding = MLP(2, 64, [32, 64])
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer_1 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.transformer_2 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head_1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.mlp_head_2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )


        self.gatenet = nn.Sequential(
            MLP(4*dim, 2, [dim, dim]),
            nn.Softmax(dim=1)
        )

        self.connect = MLP(2, dim, [dim, dim]),

        self.out = nn.Softmax(dim=1)

    def forward(self, z, q):
        # z: local feature vector       [N*Q, 4, D]
        # q: local relative coordinate  [N*Q, 4, 2]
        b, n, _ = z.shape

        # Local position embedding
        z = z + self.pos_embedding(q)          # [N*Q, 4, D]
        z = self.dropout(z)

        # gate prediction
        z_agg = rearrange(z, 'n t d -> n (t d)') # [N*Q, 4*D]
        z_gate = self.gatenet(z_agg.detach()) # [N*Q, 2]

        # index for experts
        index_1 = z_gate[:,0]>z_gate[:,1] # [N*Q, 1]
        index_2 = ~index_1 # [N*Q, 1]

        z_gate_f = self.connect(z_gate) # [N*Q, D]
        z_gate_f = torch.unsqueeze(z_gate_f, dim=1).repeat(1, 4, 1) # [N*Q,4, D]
        z = z + z_gate_f

        # select features
        z_1 = z[index_1] # [N1, 4, D]
        z_2 = z[index_2] # [N1, 4, D]

        # Attention module via Transformer
        z_1 = self.transformer_1(z_1)             # [N1, 4, D]
        z_2 = self.transformer_2(z_2)             # [N2, 4, D]
        
        # Weight estimation
        z_1 = self.mlp_head_1(z_1)                # [N1, 4, 1]
        z_2 = self.mlp_head_2(z_2)                # [N1, 4, 1]

        # concat the weight

        w = torch.empty([b, n, 1]).cuda()                # [N*Q, 4, 1]
        w[index_1] = z_1
        w[index_2] = z_2

        w = self.out(w)                           # [N*Q, 4, 1]

        return w, index_1, index_2, z_gate

class WeightTransformer2(nn.Module):
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        self.pos_embedding = nn.Linear(2, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )


    def forward(self, z, q):
        # z: local feature vector       [N*Q, 4, D]
        # q: local relative coordinate  [N*Q, 4, 2]
        b, n, _ = z.shape

        # Local position embedding
        z = z + self.pos_embedding(q)          # [N*Q, 4, D]
        z = self.dropout(z)

        # Attention module via Transformer
        z = self.transformer(z)             # [N*Q, 4, D]

        # Weight estimation
        z = self.mlp_head(z)                # [N*Q, 4, 1]
        z = torch.sigmoid(z)

        return z
    
class FeatureWeightTransformer(nn.Module):
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        self.pos_embedding = nn.Linear(2, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.out = nn.Softmax(dim=1)

    def forward(self, z, q):
        # z: local feature vector       [N*Q, 4, D]
        # q: local relative coordinate  [N*Q, 4, 2]
        b, n, _ = z.shape

        # Local position embedding
        z = z + self.pos_embedding(q)          # [N*Q, 4, D]
        z = self.dropout(z)

        # Attention module via Transformer
        z = self.transformer(z)             # [N*Q, 4, D]

        # Weight estimation
        z = self.mlp_head(z)                # [N*Q, 4, 1]
        z = self.out(z)

        return z

class FeatureAgg(nn.Module):
    def __init__(self, dim_in=64, dim_out=1, dims = [64, 128, 256]):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dims[1])
        )
        #for i in len(dims):
            
