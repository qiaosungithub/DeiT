大体idea：

经典的 classification model 都使用直接的logit head和CE loss来训练。

我们的motivation是BAR paper (文件夹下面有。) 它虽然不是classification, 但是它做的 discrete auto encoder的latent space modeling做的事情的baseline也是 logit head (经典VQVAE式的直接AR建模)，但是这个的效果随着词汇表大小的增加表现越来越差。并且inference cost是linear的：有 $N$ 个词汇就要输出 (B, N) 的logits。

BAR做的是 masked diffusion head: 只需要生成 $log_2 N$ 个 0-1 bit就可以代表预测结果，然后training就用masked diffusion loss. 这个对大的 $N$ 更robust

所以我们的思路就是实现 classification + masked diffusion head, 先在imagenet-1K上做做看能不能work。

research 目标：

先复现 DeiT 里面的 baseline, 目标是paper里面报的 81.8%。

复现之后帮我开新branch写masked diffusion head，然后跑实验调参。

你还需要ablate
1. 各种 masked diffusion head implementation
2. sampling strategy / sampling steps
3. training-time mask ratio sampling
4. maybe needs train longer
