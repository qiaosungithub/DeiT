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

    def setup(self):
        assert self.dim % self.head == 0
        self.head_dim = self.attn_dim // self.head

        self.Q = special_linear(self.attn_dim, use_bias=False)
        self.K = special_linear(self.attn_dim, use_bias=False)
        self.V = special_linear(self.attn_dim, use_bias=False)
        self.out_proj = special_linear(self.attn_dim, use_bias=False)

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

    def setup(self):
        self.attn = Attention(self.head, self.dim, self.attn_dim)
        self.ln1 = nn.LayerNorm(use_bias=False, use_scale=True, scale_init=nn.initializers.ones)
        self.mlp = nn.Sequential([
            special_linear(self.linear_dim),
            nn.gelu,
            special_linear(self.dim)
        ])
        self.ln2 = nn.LayerNorm(use_bias=False, use_scale=True, scale_init=nn.initializers.ones)
        self.learned_scale1 = self.param('learned_scale1', nn.initializers.constant(1e-4), (1,1,self.dim,))
        self.learned_scale2 = self.param('learned_scale2', nn.initializers.constant(1e-4), (1,1,self.dim,))

    def __call__(self, x,rng, training=True):
        # print('In layer: training is ', training)
        xc = x
        x = self.ln1(x)
        x = F.dropout(self.attn(x, x), rate=self.dropout_rate, training=training, rng=rng); rng = zr.next(rng)
        x = xc + F.stochastic_depth(x, self.stochastic_depth_rate, training, rng, mode='row') * self.learned_scale1; rng = zr.next(rng)

        xc = x
        x = F.dropout(self.mlp(self.ln2(x)), rate=self.dropout_rate, training=training, rng=rng); rng = zr.next(rng)
        x = xc + F.stochastic_depth(x, self.stochastic_depth_rate, training, rng, mode='row') * self.learned_scale2; rng = zr.next(rng)
        return x


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
        self.layers = [Layer(heads, embed_dim, self.linear_dim, self.attn_dim, dropout_rate=self.dropout_rate,stochastic_depth_rate=self.stochastic_depth_rate) for _ in range(n_layers)]
        self.final_ln = nn.LayerNorm(use_scale=True, use_bias=False,scale_init=nn.initializers.ones)
        self.fc = special_linear(num_classes, use_bias=True)

    def __call__(self, x:jnp.ndarray, rng, train=True):
        # print('In model: training is ', training)
        # x.shape: [B, H, W, C]
        p = self.patch_size
        x = F.patchify(x, patch_size=p) # [B, num_patch, C*patch_size**2]
        embed = self.embedding(x)
        x = jnp.concatenate((self.cls.repeat(x.shape[0], axis=0), embed), axis=1)
        # print_stat('x:', x)
        x += self.pos_emb
        x = F.dropout(x, rate=self.dropout_rate, training=train, rng=rng); rng = zr.next(rng)
        # print_stat('x:', x)

        for i,ly in enumerate(self.layers):
            x = ly(x, rng=rng, training=train); rng=zr.next(rng)
        #     print_stat(f'layer {i}:', x)
        # print_stat('x:', x)
        x = self.final_ln(x[:,0])
        return self.fc(x)

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
    attn_dim=768, # the ViT-base doesn't use the trick
    dropout_rate=0,
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