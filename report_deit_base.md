The document is for listing all differences from deit-base from our implementation.

# Attention part

- `qkv_bias` = True; `out_proj` bias = True

# Block part

- `norm_layer` all default settings (with scale and bias)

# ViT part

- `pre_nome` is set to False (应该没区别)

- pos embedding: `nn.initializers.truncated_normal(0.02)` vs `torch.randn(1, embed_len, embed_dim) * .02` + `trunc_normal_(self.pos_embed, std=.02)` (应该没区别)

- stocastic depth: `dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]` (default 0.1)

- init of cls token: `nn.init.normal_(self.cls_token, std=1e-6)` vs `self.cls = self.param('cls', nn.initializers.truncated_normal(0.02), (1, 1, embed_dim))`

- init of fc layers: (应该没区别)

```python
def init_weights_vit_timm(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()
```



