import sys, os

import zhh
from zhh.debug import print_stat, print_tensor, set_debug
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import math

from zhh.models import ModuleWrapper, TorchLinear
from functools import partial
import zhh.F as F
import zhh.random as zr

def special_linear(features,use_bias=True):
    return nn.Dense(features, kernel_init=nn.initializers.truncated_normal(0.02), bias_init=nn.initializers.zeros, use_bias=use_bias)

def sinous_embedding(l, dim):
    angles = 10000 ** (- 2 * jnp.arange(dim//2,dtype=jnp.float32) / dim)
    pos = jnp.arange(l,dtype=jnp.float32)
    # print(angles)
    mul = jnp.einsum('i,j->ij', pos, angles).reshape(1, l, -1)
    return jnp.concatenate([jnp.sin(mul), jnp.cos(mul)], axis=-1)

class Attention(nn.Module):
    head: int
    dim: int
    attn_dim: int
    use_bias: bool = False

    def setup(self):
        assert self.dim % self.head == 0
        self.head_dim = self.attn_dim // self.head

        self.Q = special_linear(self.attn_dim, use_bias=self.use_bias)
        self.K = special_linear(self.attn_dim, use_bias=self.use_bias)
        self.V = special_linear(self.attn_dim, use_bias=self.use_bias)
        self.out_proj = special_linear(self.attn_dim, use_bias=self.use_bias)

    def __call__(self, x, context):
        q = self.Q(x).reshape(*x.shape[:2], self.head, self.head_dim)
        k = self.K(context).reshape(*context.shape[:2], self.head, self.head_dim)
        v = self.V(context).reshape(*context.shape[:2], self.head, self.head_dim)

        score = jnp.einsum('bihd,bjhd->bijh', q, k) / jnp.sqrt(self.head_dim)
        score = jax.nn.softmax(score, axis=2)
        out = jnp.einsum('bijh,bjhd->bihd', score, v).reshape(*x.shape[:2], self.attn_dim)
        return self.out_proj(out)


class Layer(nn.Module):

    head: int
    dim: int
    linear_dim: int
    attn_dim: int
    dropout_rate: float
    stochastic_depth_rate: float
    use_qkv_bias: bool = False
    use_ln_bias: bool = False
    use_layer_scale: bool = True

    def setup(self):
        self.attn = Attention(self.head, self.dim, self.attn_dim, use_bias=self.use_qkv_bias)
        self.ln1 = nn.LayerNorm(use_bias=self.use_ln_bias, use_scale=True, scale_init=nn.initializers.ones)
        self.mlp = nn.Sequential([
            special_linear(self.linear_dim),
            nn.gelu,
            special_linear(self.dim)
        ])
        self.ln2 = nn.LayerNorm(use_bias=self.use_ln_bias, use_scale=True, scale_init=nn.initializers.ones)
        if self.use_layer_scale:
            self.learned_scale1 = self.param('learned_scale1', nn.initializers.constant(1e-4), (1,1,self.dim,))
            self.learned_scale2 = self.param('learned_scale2', nn.initializers.constant(1e-4), (1,1,self.dim,))

    def __call__(self, x,rng, training=True):
        # print('In layer: training is ', training)
        xc = x
        x = self.ln1(x)
        x = F.dropout(self.attn(x, x), rate=self.dropout_rate, training=training, rng=rng); rng = zr.next(rng)
        sd1 = F.stochastic_depth(x, self.stochastic_depth_rate, training, rng, mode='row')
        if self.use_layer_scale:
            x = xc + sd1 * self.learned_scale1
        else:
            x = xc + sd1
        rng = zr.next(rng)

        xc = x
        x = F.dropout(self.mlp(self.ln2(x)), rate=self.dropout_rate, training=training, rng=rng); rng = zr.next(rng)
        sd2 = F.stochastic_depth(x, self.stochastic_depth_rate, training, rng, mode='row')
        if self.use_layer_scale:
            x = xc + sd2 * self.learned_scale2
        else:
            x = xc + sd2
        rng = zr.next(rng)
        return x


# ---- Phase 2: Masked Diffusion Head ----

NUM_BITS = 10  # ceil(log2(1000)) = 10


def class_to_bits(labels, n_bits=NUM_BITS):
    """labels: (B,) int32 → (B, n_bits) int32, LSB first."""
    powers = (2 ** jnp.arange(n_bits)).astype(jnp.int32)
    return ((labels[:, None] // powers) % 2).astype(jnp.int32)


def bits_to_class(bits):
    """bits: (B, n_bits) int32 → (B,) int32."""
    powers = (2 ** jnp.arange(bits.shape[-1])).astype(jnp.int32)
    return jnp.sum(bits * powers, axis=-1)


class DiffusionLayer(nn.Module):
    """Minimal pre-norm transformer layer for the diffusion head."""
    n_heads: int
    dim: int

    def setup(self):
        self.ln1 = nn.LayerNorm(use_bias=True)
        self.attn = Attention(head=self.n_heads, dim=self.dim, attn_dim=self.dim, use_bias=True)
        self.ln2 = nn.LayerNorm(use_bias=True)
        self.mlp = nn.Sequential([
            special_linear(self.dim * 4),
            nn.gelu,
            special_linear(self.dim),
        ])

    def __call__(self, x):
        normed = self.ln1(x)
        x = x + self.attn(normed, normed)
        x = x + self.mlp(self.ln2(x))
        return x


class MaskedDiffusionHead(nn.Module):
    """
    Masked diffusion head for classification.
    Encodes label as n_bits binary, learns masked prediction via a small transformer.
    Tokens: 0 = bit-0, 1 = bit-1, 2 = MASK.
    """
    n_bits: int = NUM_BITS
    embed_dim: int = 768   # backbone CLS dim
    inner_dim: int = 256   # internal width
    n_layers: int = 2      # number of DiffusionLayer blocks (0 = no inter-bit attention)
    n_heads: int = 4
    zero_init_proj: bool = False  # zero-init out_proj kernel for training stability

    def setup(self):
        self.cls_proj = special_linear(self.inner_dim)
        self.bit_emb = nn.Embed(3, self.inner_dim)  # tokens: 0, 1, MASK=2
        self.bit_pos = self.param('bit_pos', nn.initializers.truncated_normal(0.02),
                                  (self.n_bits, self.inner_dim))
        self.diff_layers = [DiffusionLayer(n_heads=self.n_heads, dim=self.inner_dim)
                            for _ in range(self.n_layers)]
        proj_kernel_init = nn.initializers.zeros if self.zero_init_proj else nn.initializers.truncated_normal(0.02)
        self.out_proj = nn.Dense(2, kernel_init=proj_kernel_init, bias_init=nn.initializers.zeros, use_bias=True)

    def __call__(self, cls_token, masked_bits):
        """
        cls_token:   (B, embed_dim)
        masked_bits: (B, n_bits)  values in {0, 1, 2}
        Returns:     (B, n_bits, 2) logits
        """
        cls = self.cls_proj(cls_token)[:, None, :]              # (B, 1, inner_dim)
        bits = self.bit_emb(masked_bits) + self.bit_pos         # (B, n_bits, inner_dim)
        x = jnp.concatenate([cls, bits], axis=1)                # (B, 1+n_bits, inner_dim)
        for layer in self.diff_layers:
            x = layer(x)
        return self.out_proj(x[:, 1:, :])                       # (B, n_bits, 2)


class MLPDiffusionHead(nn.Module):
    """
    MLP baseline for masked diffusion head.
    Flattens CLS + all bit embeddings into a single vector, passes through
    dense layers. No inter-bit attention — tests whether attention buys
    anything over a plain MLP for 10 bits.
    """
    n_bits: int = NUM_BITS
    embed_dim: int = 768
    inner_dim: int = 256
    n_layers: int = 2   # number of hidden MLP layers
    zero_init_proj: bool = False  # zero-init out_proj kernel for training stability

    def setup(self):
        self.cls_proj = special_linear(self.inner_dim)
        self.bit_emb = nn.Embed(3, self.inner_dim)           # tokens: 0, 1, MASK=2
        self.bit_pos = self.param('bit_pos', nn.initializers.truncated_normal(0.02),
                                  (self.n_bits, self.inner_dim))
        hidden_dim = self.inner_dim * 4                      # 256*4=1024
        self.hidden_layers = [special_linear(hidden_dim) for _ in range(self.n_layers)]
        proj_kernel_init = nn.initializers.zeros if self.zero_init_proj else nn.initializers.truncated_normal(0.02)
        self.out_proj = nn.Dense(self.n_bits * 2, kernel_init=proj_kernel_init, bias_init=nn.initializers.zeros, use_bias=True)

    def __call__(self, cls_token, masked_bits):
        """
        cls_token:   (B, embed_dim)
        masked_bits: (B, n_bits)  values in {0, 1, 2}
        Returns:     (B, n_bits, 2) logits
        """
        cls = self.cls_proj(cls_token)                       # (B, inner_dim)
        bits = self.bit_emb(masked_bits) + self.bit_pos      # (B, n_bits, inner_dim)
        bits_flat = bits.reshape(bits.shape[0], -1)          # (B, n_bits*inner_dim)
        x = jnp.concatenate([cls, bits_flat], axis=-1)       # (B, (n_bits+1)*inner_dim)
        for fc in self.hidden_layers:
            x = nn.gelu(fc(x))
        x = self.out_proj(x)                                 # (B, n_bits*2)
        return x.reshape(x.shape[0], self.n_bits, 2)         # (B, n_bits, 2)


class ViT(nn.Module):

    channels: int
    image_size: int
    patch_size: int
    num_classes: int
    embed_dim: int
    n_layers: int
    heads: int
    linear_dim: int
    attn_dim: int
    dropout_rate: float
    stochastic_depth_rate: float
    dtype: jnp.dtype = jnp.float32
    use_qkv_bias: bool = False
    use_ln_bias: bool = False
    use_layer_scale: bool = True
    # Phase 2: masked diffusion head
    use_diffusion_head: bool = False
    n_bits: int = 10
    head_inner_dim: int = 256
    head_n_layers: int = 2
    head_n_heads: int = 4
    head_type: str = 'attention'   # 'attention' | 'mlp'
    head_zero_init_proj: bool = False  # zero-init final projection layer
    head_aux_ce: bool = False         # add auxiliary CE head alongside diffusion head

    def setup(self):
        image_size = self.image_size
        patch_size = self.patch_size
        num_classes = self.num_classes
        embed_dim = self.embed_dim
        n_layers = self.n_layers
        heads = self.heads

        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2

        # modules
        self.embedding = TorchLinear(self.channels * (patch_size ** 2), embed_dim)
        # self.pos_emb = sinous_embedding(num_patches + 1, embed_dim)
        self.pos_emb = self.param('pos_emb', nn.initializers.truncated_normal(0.02), (1, num_patches + 1, embed_dim))
        self.cls = self.param('cls', nn.initializers.truncated_normal(0.02), (1, 1, embed_dim))
        self.layers = [Layer(heads, embed_dim, self.linear_dim, self.attn_dim,
                             dropout_rate=self.dropout_rate,
                             stochastic_depth_rate=self.stochastic_depth_rate,
                             use_qkv_bias=self.use_qkv_bias,
                             use_ln_bias=self.use_ln_bias,
                             use_layer_scale=self.use_layer_scale)
                       for _ in range(n_layers)]
        self.final_ln = nn.LayerNorm(use_scale=True, use_bias=False,scale_init=nn.initializers.ones)
        if not self.use_diffusion_head:
            self.fc = special_linear(num_classes, use_bias=True)
        elif self.head_type == 'mlp':
            self.diffusion_head = MLPDiffusionHead(
                n_bits=self.n_bits,
                embed_dim=embed_dim,
                inner_dim=self.head_inner_dim,
                n_layers=self.head_n_layers,
                zero_init_proj=self.head_zero_init_proj,
            )
            if self.head_aux_ce:
                self.fc = special_linear(num_classes, use_bias=True)
        else:
            self.diffusion_head = MaskedDiffusionHead(
                n_bits=self.n_bits,
                embed_dim=embed_dim,
                inner_dim=self.head_inner_dim,
                n_layers=self.head_n_layers,
                n_heads=self.head_n_heads,
                zero_init_proj=self.head_zero_init_proj,
            )
            if self.head_aux_ce:
                self.fc = special_linear(num_classes, use_bias=True)

    def encode(self, x: jnp.ndarray, rng, train=True):
        """Return CLS token embedding (before head)."""
        p = self.patch_size
        x = F.patchify(x, patch_size=p)
        embed = self.embedding(x)
        x = jnp.concatenate((self.cls.repeat(x.shape[0], axis=0), embed), axis=1)
        x += self.pos_emb
        x = F.dropout(x, rate=self.dropout_rate, training=train, rng=rng); rng = zr.next(rng)
        for ly in self.layers:
            x = ly(x, rng=rng, training=train); rng = zr.next(rng)
        return self.final_ln(x[:, 0])  # (B, embed_dim)

    def __call__(self, x:jnp.ndarray, rng, train=True, masked_bits=None, return_aux_ce=False):
        # x.shape: [B, H, W, C]
        cls = self.encode(x, rng, train)
        if not self.use_diffusion_head:
            return self.fc(cls)
        else:
            if masked_bits is None:
                masked_bits = jnp.full((x.shape[0], self.n_bits), 2)
            diff_logits = self.diffusion_head(cls, masked_bits)
            if return_aux_ce and self.head_aux_ce:
                return diff_logits, self.fc(cls)
            return diff_logits

ViT_base = partial(
    ViT,
    channels=3,
    image_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    n_layers=12,
    heads=12,
    linear_dim=3072,
    attn_dim=768,
    dropout_rate=0,
    use_qkv_bias=False,
    use_ln_bias=False,
    use_layer_scale=True,
)

# Exact match to reference DeiT-B: biases everywhere, no LearnedScale
ViT_base_v2 = partial(
    ViT,
    channels=3,
    image_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    n_layers=12,
    heads=12,
    linear_dim=3072,
    attn_dim=768,
    dropout_rate=0,
    use_qkv_bias=True,
    use_ln_bias=True,
    use_layer_scale=False,
)

# Biases + LearnedScale (intermediate variant)
ViT_base_v3 = partial(
    ViT,
    channels=3,
    image_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    n_layers=12,
    heads=12,
    linear_dim=3072,
    attn_dim=768,
    dropout_rate=0,
    use_qkv_bias=True,
    use_ln_bias=True,
    use_layer_scale=True,
)

# Phase 2: DeiT-B backbone (v3) + masked diffusion head
ViT_base_mdh = partial(
    ViT,
    channels=3,
    image_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    n_layers=12,
    heads=12,
    linear_dim=3072,
    attn_dim=768,
    dropout_rate=0,
    use_qkv_bias=True,
    use_ln_bias=True,
    use_layer_scale=True,
    use_diffusion_head=True,
    n_bits=NUM_BITS,
    head_inner_dim=256,
    head_n_layers=2,
    head_n_heads=4,
)

# Phase 2 Run E: zero-init out_proj variant
ViT_base_mdh_zero_init = partial(
    ViT,
    channels=3,
    image_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    n_layers=12,
    heads=12,
    linear_dim=3072,
    attn_dim=768,
    dropout_rate=0,
    use_qkv_bias=True,
    use_ln_bias=True,
    use_layer_scale=True,
    use_diffusion_head=True,
    n_bits=NUM_BITS,
    head_inner_dim=256,
    head_n_layers=2,
    head_n_heads=4,
    head_zero_init_proj=True,
)

# Phase 2: MLP baseline — same backbone as mdh, MLP head instead of attention
ViT_base_mdh_mlp = partial(
    ViT,
    channels=3,
    image_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    n_layers=12,
    heads=12,
    linear_dim=3072,
    attn_dim=768,
    dropout_rate=0,
    use_qkv_bias=True,
    use_ln_bias=True,
    use_layer_scale=True,
    use_diffusion_head=True,
    head_type='mlp',
    n_bits=NUM_BITS,
    head_inner_dim=256,
    head_n_layers=2,
)

# Phase 2: larger attention head — inner_dim=512, n_layers=4 (more capacity)
ViT_base_mdh_large = partial(
    ViT,
    channels=3,
    image_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    n_layers=12,
    heads=12,
    linear_dim=3072,
    attn_dim=768,
    dropout_rate=0,
    use_qkv_bias=True,
    use_ln_bias=True,
    use_layer_scale=True,
    use_diffusion_head=True,
    n_bits=NUM_BITS,
    head_inner_dim=512,
    head_n_layers=4,
    head_n_heads=8,
)

# Phase 2 Run H: attention head + auxiliary CE loss (λ=0.1 by default)
ViT_base_mdh_aux_ce = partial(
    ViT,
    channels=3,
    image_size=224,
    patch_size=16,
    num_classes=1000,
    embed_dim=768,
    n_layers=12,
    heads=12,
    linear_dim=3072,
    attn_dim=768,
    dropout_rate=0,
    use_qkv_bias=True,
    use_ln_bias=True,
    use_layer_scale=True,
    use_diffusion_head=True,
    n_bits=NUM_BITS,
    head_inner_dim=256,
    head_n_layers=2,
    head_n_heads=4,
    head_aux_ce=True,
)

ViT_debug = partial(
    ViT,
    channels=3,
    image_size=224,
    patch_size=112,
    num_classes=1000,
    embed_dim=4,
    n_layers=1,
    heads=2,
    linear_dim=4,
    attn_dim=4,
    dropout_rate=0,
)

if __name__ == '__main__':
    set_debug()
    model = ModuleWrapper(
    # ViT(
    #     channels=3,
    #     image_size=224,
    #     patch_size=16,
    #     num_classes=7,
    #     embed_dim=8,
    #     n_layers=1,
    #     heads=2,
    #     linear_dim=8,
    #     attn_dim=8,
    #     dropout_rate=0.1
    # )
    ViT_debug(stochastic_depth_rate=0.1),
    optimizer=optax.adam(0.001))
    print('-'*10)
    model.step(jnp.zeros((5,224,224,3)), update=False)
    print('-'*10)
    print(model.num_parameters())  # This should be 86M
    # print(model)
    # 
    # class Model(nn.Module):
    #     def setup(self):
    #         self.foo = nn.Dense(100, kernel_init=torch_weight_initializer, bias_init=torch_bias_initializer(100))
    #     def __call__(self, x):
    #         return self.foo(x)
    # model = ModuleWrapper(Model(), optimizer=optax.adam(0.001))
    # model.step(jnp.zeros((100,100)), update=False)
    # # print(model._state.params)
    # weight = model._state.params['foo']['kernel']
    # bias = model._state.params['foo']['bias']
    # print_stat('weight:', weight)
    # print_stat('bias:', bias)