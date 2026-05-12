## 2026-05-10 11:24:36 UTC
deit_windows=9
6380	sanity-run: full alignment with reference (WD mask, b2=0.999, LR schedule, cls init, stochastic depth schedule, final LN bias) — target 81.8	v6e-8-spot-gzy-favaxa	[1;32mRunning[0m (ep=133.88)
6381	phase2 Run H: attention head + aux CE loss (lambda=0.1). Tests whether CE supervision on backbone improves convergence speed vs pure diffusi	v6e-8-spot-gzy-qxxa8y	[1;32mRunning[0m (ep=136)
6382	phase2 Run E: zero-init out_proj in diffusion head; otherwise same as Run C (uniform schedule, biases+LS backbone). Tests whether zero-init 	v6e-8-spot-gzy-axuxm0	[1;32mRunning[0m (ep=134)
6383	phase2 Run G: larger diffusion head (512-dim, 4 layers, 8 heads) vs Run C (256-dim, 2 layers). Ablates head capacity: does more capacity hel	v6e-8-spot-gzy-06q7u9	[1;32mRunning[0m (ep=132.36)
6388	phase2 Run I: warm-start diffusion head from Phase 1 Run 3 (73.14% CE backbone). Tests whether pretrained backbone accelerates convergence v	v6e-8-spot-gzy-j3rqvs	[1;32mRunning[0m (ep=129)
6389	deit phase2 Run L: cross-attention head over full patch tokens (4 layers) + K10 multi-mask	v6e-8-spot-gzy-p1u4mx	[1;33munknown[0m (resumed in 6398)
6390	deit phase2 Run M: tiny-bit MLP (bit_dim=8, hidden=3072) + K10 multi-mask	v6e-8-spot-gzy-cz2ivo	[1;32mRunning[0m (ep=85.12)
6392	deit phase2 Run N: tiny-bit MLP + K10 + low-aug mild (no RA, reprob0, repeated1, wd0.02, sd0.05)	v6e-8-spot-gzy-3djlis	[1;32mRunning[0m (ep=106)
6398	deit phase2 Run L: cross-attention head over full patch tokens (4 layers) + K10 multi-mask; resume:6389; stage:2	v6e-8-spot-gzy-z169mq	[1;31mError[0m

## 2026-05-10 11:25:06 UTC
deit_windows=9
6380	sanity-run: full alignment with reference (WD mask, b2=0.999, LR schedule, cls init, stochastic depth schedule, final LN bias) — target 81.8	v6e-8-spot-gzy-favaxa	[1;32mRunning[0m (ep=134)
6381	phase2 Run H: attention head + aux CE loss (lambda=0.1). Tests whether CE supervision on backbone improves convergence speed vs pure diffusi	v6e-8-spot-gzy-qxxa8y	[1;32mRunning[0m (ep=136)
6382	phase2 Run E: zero-init out_proj in diffusion head; otherwise same as Run C (uniform schedule, biases+LS backbone). Tests whether zero-init 	v6e-8-spot-gzy-axuxm0	[1;32mRunning[0m (ep=134)
6383	phase2 Run G: larger diffusion head (512-dim, 4 layers, 8 heads) vs Run C (256-dim, 2 layers). Ablates head capacity: does more capacity hel	v6e-8-spot-gzy-06q7u9	[1;32mRunning[0m (ep=133)
6388	phase2 Run I: warm-start diffusion head from Phase 1 Run 3 (73.14% CE backbone). Tests whether pretrained backbone accelerates convergence v	v6e-8-spot-gzy-j3rqvs	[1;32mRunning[0m (ep=129)
6389	deit phase2 Run L: cross-attention head over full patch tokens (4 layers) + K10 multi-mask	v6e-8-spot-gzy-p1u4mx	[1;33munknown[0m (resumed in 6398)
6390	deit phase2 Run M: tiny-bit MLP (bit_dim=8, hidden=3072) + K10 multi-mask	v6e-8-spot-gzy-cz2ivo	[1;32mRunning[0m (ep=85.20)
6392	deit phase2 Run N: tiny-bit MLP + K10 + low-aug mild (no RA, reprob0, repeated1, wd0.02, sd0.05)	v6e-8-spot-gzy-3djlis	[1;32mRunning[0m (ep=106)
6398	deit phase2 Run L: cross-attention head over full patch tokens (4 layers) + K10 multi-mask; resume:6389; stage:2	v6e-8-spot-gzy-z169mq	[1;31mError[0m

## 2026-05-10 11:38:08 UTC
deit_windows=8
6380	sanity-run: full alignment with reference (WD mask, b2=0.999, LR schedule, cls init, stochastic depth schedule, final LN bias) — target 81.8	v6e-8-spot-gzy-favaxa	[1;32mRunning[0m (ep=136.04)
6381	phase2 Run H: attention head + aux CE loss (lambda=0.1). Tests whether CE supervision on backbone improves convergence speed vs pure diffusi	v6e-8-spot-gzy-qxxa8y	[1;32mRunning[0m (ep=138.03)
6382	phase2 Run E: zero-init out_proj in diffusion head; otherwise same as Run C (uniform schedule, biases+LS backbone). Tests whether zero-init 	v6e-8-spot-gzy-axuxm0	[1;32mRunning[0m (ep=136)
6383	phase2 Run G: larger diffusion head (512-dim, 4 layers, 8 heads) vs Run C (256-dim, 2 layers). Ablates head capacity: does more capacity hel	v6e-8-spot-gzy-06q7u9	[1;32mRunning[0m (ep=135)
6388	phase2 Run I: warm-start diffusion head from Phase 1 Run 3 (73.14% CE backbone). Tests whether pretrained backbone accelerates convergence v	v6e-8-spot-gzy-j3rqvs	[1;32mRunning[0m (ep=131)
6390	deit phase2 Run M: tiny-bit MLP (bit_dim=8, hidden=3072) + K10 multi-mask	v6e-8-spot-gzy-cz2ivo	[1;32mRunning[0m (ep=87.36)
6392	deit phase2 Run N: tiny-bit MLP + K10 + low-aug mild (no RA, reprob0, repeated1, wd0.02, sd0.05)	v6e-8-spot-gzy-3djlis	[1;32mRunning[0m (ep=108.46)
6400	deit phase2 Run L: cross-attention head over full patch tokens (4 layers) + K10 multi-mask	v6e-8-spot-gzy-z169mq	[1;33mUnknown[0m

## 2026-05-10 12:08:11 UTC
deit_windows=8
6380	sanity-run: full alignment with reference (WD mask, b2=0.999, LR schedule, cls init, stochastic depth schedule, final LN bias) — target 81.8	v6e-8-spot-gzy-favaxa	[1;32mRunning[0m (ep=140.75)
6381	phase2 Run H: attention head + aux CE loss (lambda=0.1). Tests whether CE supervision on backbone improves convergence speed vs pure diffusi	v6e-8-spot-gzy-qxxa8y	[1;32mRunning[0m (ep=143)
6382	phase2 Run E: zero-init out_proj in diffusion head; otherwise same as Run C (uniform schedule, biases+LS backbone). Tests whether zero-init 	v6e-8-spot-gzy-axuxm0	[1;32mRunning[0m (ep=141)
6383	phase2 Run G: larger diffusion head (512-dim, 4 layers, 8 heads) vs Run C (256-dim, 2 layers). Ablates head capacity: does more capacity hel	v6e-8-spot-gzy-06q7u9	[1;32mRunning[0m (ep=139)
6388	phase2 Run I: warm-start diffusion head from Phase 1 Run 3 (73.14% CE backbone). Tests whether pretrained backbone accelerates convergence v	v6e-8-spot-gzy-j3rqvs	[1;32mRunning[0m (ep=136)
6390	deit phase2 Run M: tiny-bit MLP (bit_dim=8, hidden=3072) + K10 multi-mask	v6e-8-spot-gzy-cz2ivo	[1;32mRunning[0m (ep=92.31)
6392	deit phase2 Run N: tiny-bit MLP + K10 + low-aug mild (no RA, reprob0, repeated1, wd0.02, sd0.05)	v6e-8-spot-gzy-3djlis	[1;32mRunning[0m (ep=115)
6404	deit phase2 Run L: cross-attention head over full patch tokens (4 layers) + K10 multi-mask	v6e-8-spot-gzy-z169mq	[1;33mUnknown[0m

## 2026-05-10 12:38:13 UTC
deit_windows=8
6380	sanity-run: full alignment with reference (WD mask, b2=0.999, LR schedule, cls init, stochastic depth schedule, final LN bias) — target 81.8	v6e-8-spot-gzy-favaxa	[1;32mRunning[0m (ep=145.55)
6381	phase2 Run H: attention head + aux CE loss (lambda=0.1). Tests whether CE supervision on backbone improves convergence speed vs pure diffusi	v6e-8-spot-gzy-qxxa8y	[1;32mRunning[0m (ep=148)
6382	phase2 Run E: zero-init out_proj in diffusion head; otherwise same as Run C (uniform schedule, biases+LS backbone). Tests whether zero-init 	v6e-8-spot-gzy-axuxm0	[1;32mRunning[0m (ep=146)
6383	phase2 Run G: larger diffusion head (512-dim, 4 layers, 8 heads) vs Run C (256-dim, 2 layers). Ablates head capacity: does more capacity hel	v6e-8-spot-gzy-06q7u9	[1;32mRunning[0m (ep=144.19)
6388	phase2 Run I: warm-start diffusion head from Phase 1 Run 3 (73.14% CE backbone). Tests whether pretrained backbone accelerates convergence v	v6e-8-spot-gzy-j3rqvs	[1;32mRunning[0m (ep=141)
6390	deit phase2 Run M: tiny-bit MLP (bit_dim=8, hidden=3072) + K10 multi-mask	v6e-8-spot-gzy-cz2ivo	[1;32mRunning[0m (ep=97.27)
6392	deit phase2 Run N: tiny-bit MLP + K10 + low-aug mild (no RA, reprob0, repeated1, wd0.02, sd0.05)	v6e-8-spot-gzy-3djlis	[1;32mRunning[0m (ep=121)
6405	deit phase2 Run L: cross-attention head over full patch tokens (4 layers) + K10 multi-mask	v6e-8-spot-gzy-z169mq	[1;32mRunning[0m (ep=2.40)

## 2026-05-10 12:43:11 UTC
deit_windows=8
6380	sanity-run: full alignment with reference (WD mask, b2=0.999, LR schedule, cls init, stochastic depth schedule, final LN bias) — target 81.8	v6e-8-spot-gzy-favaxa	[1;32mRunning[0m (ep=146.35)
6381	phase2 Run H: attention head + aux CE loss (lambda=0.1). Tests whether CE supervision on backbone improves convergence speed vs pure diffusi	v6e-8-spot-gzy-qxxa8y	[1;32mRunning[0m (ep=149)
6382	phase2 Run E: zero-init out_proj in diffusion head; otherwise same as Run C (uniform schedule, biases+LS backbone). Tests whether zero-init 	v6e-8-spot-gzy-axuxm0	[1;32mRunning[0m (ep=146)
6383	phase2 Run G: larger diffusion head (512-dim, 4 layers, 8 heads) vs Run C (256-dim, 2 layers). Ablates head capacity: does more capacity hel	v6e-8-spot-gzy-06q7u9	[1;32mRunning[0m (ep=145)
6388	phase2 Run I: warm-start diffusion head from Phase 1 Run 3 (73.14% CE backbone). Tests whether pretrained backbone accelerates convergence v	v6e-8-spot-gzy-j3rqvs	[1;32mRunning[0m (ep=141)
6390	deit phase2 Run M: tiny-bit MLP (bit_dim=8, hidden=3072) + K10 multi-mask	v6e-8-spot-gzy-cz2ivo	[1;32mRunning[0m (ep=98.07)
6392	deit phase2 Run N: tiny-bit MLP + K10 + low-aug mild (no RA, reprob0, repeated1, wd0.02, sd0.05)	v6e-8-spot-gzy-3djlis	[1;32mRunning[0m (ep=122)
6405	deit phase2 Run L: cross-attention head over full patch tokens (4 layers) + K10 multi-mask	v6e-8-spot-gzy-z169mq	[1;33mUnknown[0m

### Latest eval snapshot
### window=6380 tpu=v6e-8-spot-gzy-favaxa
tag=sanity-run: full alignment with reference (WD mask, b2=0.999, LR schedule, cls init, stochastic depth schedule, final LN bias) — target 81.8
status=[1;32mRunning[0m (ep=146.35)
logdir=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260509_212142_mxfpdz_kmh-tpuvm-v6e-8-spot-gzy-favaxa_asia-northeast1-b__b_lr_ep_eval
I0510 05:45:54.834356 138353402066944 logging_util.py:18] eval epoch: 79, loss: 1.957492, accuracy: 59.342000
I0510 05:45:54.834495 138353402066944 logging_util.py:133] [100080] eval_loss=1.9575, eval_accuracy=0.59342, ep=79.99
I0510 07:51:31.791923 138353402066944 logging_util.py:18] eval epoch: 99, loss: 1.774279, accuracy: 62.754000
I0510 07:51:31.792053 138353402066944 logging_util.py:133] [125100] eval_loss=1.7743, eval_accuracy=0.62754, ep=99.988
I0510 09:56:59.650715 138353402066944 logging_util.py:18] eval epoch: 119, loss: 1.834525, accuracy: 62.622000
I0510 09:56:59.650856 138353402066944 logging_util.py:133] [150120] eval_loss=1.8345, eval_accuracy=0.62622, ep=119.99
I0510 12:02:35.289051 138353402066944 logging_util.py:18] eval epoch: 139, loss: 1.834445, accuracy: 63.356000
I0510 12:02:35.289182 138353402066944 logging_util.py:133] [175140] eval_loss=1.8344, eval_accuracy=0.63356, ep=139.98

### window=6381 tpu=v6e-8-spot-gzy-qxxa8y
tag=phase2 Run H: attention head + aux CE loss (lambda=0.1). Tests whether CE supervision on backbone improves convergence speed vs pure diffusi
status=[1;32mRunning[0m (ep=149)
logdir=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260509_213033_is69hk_kmh-tpuvm-v6e-8-spot-gzy-qxxa8y_asia-northeast1-b__b_lr_ep_eval
I0510 07:42:26.887118 131896659294208 logging_util.py:18] eval epoch: 99, accuracy_iter: 53.334000, invalid_rate: 0.900000
I0510 07:42:26.887233 131896659294208 logging_util.py:133] [125100] eval_loss=0.45417, eval_accuracy=0.50792, eval_accuracy_iter=0.53334, eval_invalid_rate=0.009, eval_invalid_iter_rate=0.00022, ep=99.988
I0510 09:44:33.487574 131896659294208 logging_util.py:18] eval epoch: 119, loss: 0.437169, accuracy: 51.548000
I0510 09:44:33.487690 131896659294208 logging_util.py:18] eval epoch: 119, accuracy_iter: 53.886000, invalid_rate: 0.646000
I0510 09:44:33.487797 131896659294208 logging_util.py:133] [150120] eval_loss=0.43717, eval_accuracy=0.51548, eval_accuracy_iter=0.53886, eval_invalid_rate=0.00646, eval_invalid_iter_rate=0.00026, ep=119.99
I0510 11:46:41.752510 131896659294208 logging_util.py:18] eval epoch: 139, loss: 0.437972, accuracy: 52.506000
I0510 11:46:41.752627 131896659294208 logging_util.py:18] eval epoch: 139, accuracy_iter: 54.816000, invalid_rate: 0.378000
I0510 11:46:41.752727 131896659294208 logging_util.py:133] [175140] eval_loss=0.43797, eval_accuracy=0.52506, eval_accuracy_iter=0.54816, eval_invalid_rate=0.00378, eval_invalid_iter_rate=0.00016, ep=139.98

### window=6382 tpu=v6e-8-spot-gzy-axuxm0
tag=phase2 Run E: zero-init out_proj in diffusion head; otherwise same as Run C (uniform schedule, biases+LS backbone). Tests whether zero-init 
status=[1;32mRunning[0m (ep=146)
logdir=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260509_214004_nin5vh_kmh-tpuvm-v6e-8-spot-gzy-axuxm0_asia-northeast1-b__b_lr_ep_eval
I0510 07:55:13.494560 134033742792704 logging_util.py:18] eval epoch: 99, accuracy_iter: 1.016000, invalid_rate: 1.724000
I0510 07:55:13.494697 134033742792704 logging_util.py:133] [125100] eval_loss=0.67832, eval_accuracy=0.00426, eval_accuracy_iter=0.01016, eval_invalid_rate=0.01724, eval_invalid_iter_rate=0.00042, ep=99.988
I0510 09:57:22.967946 134033742792704 logging_util.py:18] eval epoch: 119, loss: 0.669255, accuracy: 0.806000
I0510 09:57:22.968081 134033742792704 logging_util.py:18] eval epoch: 119, accuracy_iter: 1.874000, invalid_rate: 2.144000
I0510 09:57:22.968196 134033742792704 logging_util.py:133] [150120] eval_loss=0.66925, eval_accuracy=0.00806, eval_accuracy_iter=0.01874, eval_invalid_rate=0.02144, eval_invalid_iter_rate=6e-05, ep=119.99
I0510 11:59:48.943911 134033742792704 logging_util.py:18] eval epoch: 139, loss: 0.659177, accuracy: 1.326000
I0510 11:59:48.944211 134033742792704 logging_util.py:18] eval epoch: 139, accuracy_iter: 3.172000, invalid_rate: 3.734000
I0510 11:59:48.944433 134033742792704 logging_util.py:133] [175140] eval_loss=0.65918, eval_accuracy=0.01326, eval_accuracy_iter=0.03172, eval_invalid_rate=0.03734, eval_invalid_iter_rate=0.00022, ep=139.98

### window=6383 tpu=v6e-8-spot-gzy-06q7u9
tag=phase2 Run G: larger diffusion head (512-dim, 4 layers, 8 heads) vs Run C (256-dim, 2 layers). Ablates head capacity: does more capacity hel
status=[1;32mRunning[0m (ep=145)
logdir=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260509_214224_qa2gek_kmh-tpuvm-v6e-8-spot-gzy-06q7u9_asia-northeast1-b__b_lr_ep_eval
I0510 08:00:28.689179 126804566546432 logging_util.py:18] eval epoch: 99, accuracy_iter: 2.658000, invalid_rate: 1.550000
I0510 08:00:28.689292 126804566546432 logging_util.py:133] [125100] eval_loss=0.66598, eval_accuracy=0.01044, eval_accuracy_iter=0.02658, eval_invalid_rate=0.0155, eval_invalid_iter_rate=4e-05, ep=99.988
I0510 10:04:26.730893 126804566546432 logging_util.py:18] eval epoch: 119, loss: 0.657741, accuracy: 1.850000
I0510 10:04:26.731019 126804566546432 logging_util.py:18] eval epoch: 119, accuracy_iter: 4.320000, invalid_rate: 3.918000
I0510 10:04:26.731139 126804566546432 logging_util.py:133] [150120] eval_loss=0.65774, eval_accuracy=0.0185, eval_accuracy_iter=0.0432, eval_invalid_rate=0.03918, eval_invalid_iter_rate=2e-05, ep=119.99
I0510 12:08:27.774757 126804566546432 logging_util.py:18] eval epoch: 139, loss: 0.644498, accuracy: 2.952000
I0510 12:08:27.774875 126804566546432 logging_util.py:18] eval epoch: 139, accuracy_iter: 6.636000, invalid_rate: 2.694000
I0510 12:08:27.774989 126804566546432 logging_util.py:133] [175140] eval_loss=0.6445, eval_accuracy=0.02952, eval_accuracy_iter=0.06636, eval_invalid_rate=0.02694, eval_invalid_iter_rate=0, ep=139.98

### window=6388 tpu=v6e-8-spot-gzy-j3rqvs
tag=phase2 Run I: warm-start diffusion head from Phase 1 Run 3 (73.14% CE backbone). Tests whether pretrained backbone accelerates convergence v
status=[1;32mRunning[0m (ep=141)
logdir=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260509_221135_gmkq7q_kmh-tpuvm-v6e-8-spot-gzy-j3rqvs_asia-northeast1-b__b_lr_ep_eval
I0510 08:25:42.097741 125830992791552 logging_util.py:18] eval epoch: 99, accuracy_iter: 59.288000, invalid_rate: 1.226000
I0510 08:25:42.097844 125830992791552 logging_util.py:133] [125100] eval_loss=0.34528, eval_accuracy=0.55438, eval_accuracy_iter=0.59288, eval_invalid_rate=0.01226, eval_invalid_iter_rate=0.0002, ep=99.988
I0510 10:27:42.504707 125830992791552 logging_util.py:18] eval epoch: 119, loss: 0.343700, accuracy: 57.410000
I0510 10:27:42.504832 125830992791552 logging_util.py:18] eval epoch: 119, accuracy_iter: 60.872000, invalid_rate: 0.686000
I0510 10:27:42.504936 125830992791552 logging_util.py:133] [150120] eval_loss=0.3437, eval_accuracy=0.5741, eval_accuracy_iter=0.60872, eval_invalid_rate=0.00686, eval_invalid_iter_rate=0.00022, ep=119.99
I0510 12:29:41.421755 125830992791552 logging_util.py:18] eval epoch: 139, loss: 0.334276, accuracy: 58.986000
I0510 12:29:41.421875 125830992791552 logging_util.py:18] eval epoch: 139, accuracy_iter: 62.470000, invalid_rate: 0.444000
I0510 12:29:41.421976 125830992791552 logging_util.py:133] [175140] eval_loss=0.33428, eval_accuracy=0.58986, eval_accuracy_iter=0.6247, eval_invalid_rate=0.00444, eval_invalid_iter_rate=0.00024, ep=139.98

### window=6390 tpu=v6e-8-spot-gzy-cz2ivo
tag=deit phase2 Run M: tiny-bit MLP (bit_dim=8, hidden=3072) + K10 multi-mask
status=[1;32mRunning[0m (ep=98.07)
logdir=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_024012_fw1rbv_kmh-tpuvm-v6e-8-spot-gzy-cz2ivo_us-east5-b__b_lr_ep_eval
I0510 06:46:02.435327 130698839156736 logging_util.py:18] eval epoch: 39, accuracy_iter: 47.260000, invalid_rate: 1.226000
I0510 06:46:02.435436 130698839156736 logging_util.py:133] [50040] eval_loss=0.43096, eval_accuracy=0.4006, eval_accuracy_iter=0.4726, eval_invalid_rate=0.01226, eval_invalid_iter_rate=0.00018, ep=39.995
I0510 08:47:48.584784 130698839156736 logging_util.py:18] eval epoch: 59, loss: 0.399573, accuracy: 48.126000
I0510 08:47:48.584914 130698839156736 logging_util.py:18] eval epoch: 59, accuracy_iter: 54.348000, invalid_rate: 0.744000
I0510 08:47:48.585018 130698839156736 logging_util.py:133] [75060] eval_loss=0.39957, eval_accuracy=0.48126, eval_accuracy_iter=0.54348, eval_invalid_rate=0.00744, eval_invalid_iter_rate=0.00016, ep=59.993
I0510 10:49:41.939933 130698839156736 logging_util.py:18] eval epoch: 79, loss: 0.411026, accuracy: 51.186000
I0510 10:49:41.940230 130698839156736 logging_util.py:18] eval epoch: 79, accuracy_iter: 56.400000, invalid_rate: 0.660000
I0510 10:49:41.940362 130698839156736 logging_util.py:133] [100080] eval_loss=0.41103, eval_accuracy=0.51186, eval_accuracy_iter=0.564, eval_invalid_rate=0.0066, eval_invalid_iter_rate=0.00028, ep=79.99

### window=6392 tpu=v6e-8-spot-gzy-3djlis
tag=deit phase2 Run N: tiny-bit MLP + K10 + low-aug mild (no RA, reprob0, repeated1, wd0.02, sd0.05)
status=[1;32mRunning[0m (ep=122)
logdir=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_024337_vt4mt2_kmh-tpuvm-v6e-8-spot-gzy-3djlis_asia-northeast1-b__b_lr_ep_eval
I0510 09:16:57.273761 126447346067456 logging_util.py:18] eval epoch: 79, accuracy_iter: 60.954000, invalid_rate: 0.696000
I0510 09:16:57.273892 126447346067456 logging_util.py:133] [100080] eval_loss=0.39084, eval_accuracy=0.56896, eval_accuracy_iter=0.60954, eval_invalid_rate=0.00696, eval_invalid_iter_rate=4e-05, ep=79.99
I0510 10:54:51.424077 126447346067456 logging_util.py:18] eval epoch: 99, loss: 0.414502, accuracy: 57.566000
I0510 10:54:51.424340 126447346067456 logging_util.py:18] eval epoch: 99, accuracy_iter: 61.624000, invalid_rate: 0.826000
I0510 10:54:51.424509 126447346067456 logging_util.py:133] [125100] eval_loss=0.4145, eval_accuracy=0.57566, eval_accuracy_iter=0.61624, eval_invalid_rate=0.00826, eval_invalid_iter_rate=0.00018, ep=99.988
I0510 12:33:04.162859 126447346067456 logging_util.py:18] eval epoch: 119, loss: 0.420070, accuracy: 59.060000
I0510 12:33:04.162986 126447346067456 logging_util.py:18] eval epoch: 119, accuracy_iter: 62.800000, invalid_rate: 0.722000
I0510 12:33:04.163091 126447346067456 logging_util.py:133] [150120] eval_loss=0.42007, eval_accuracy=0.5906, eval_accuracy_iter=0.628, eval_invalid_rate=0.00722, eval_invalid_iter_rate=0.0001, ep=119.99

### window=6405 tpu=v6e-8-spot-gzy-z169mq
tag=deit phase2 Run L: cross-attention head over full patch tokens (4 layers) + K10 multi-mask
status=[1;33mUnknown[0m
logdir=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_121100_lkmvun_kmh-tpuvm-v6e-8-spot-gzy-z169mq_asia-northeast1-b__b_lr_ep_eval
I0510 12:25:55.981643 124672498300928 logging_util.py:18] eval epoch: 0, loss: 0.692041, accuracy: 0.100000
I0510 12:25:55.981736 124672498300928 logging_util.py:18] eval epoch: 0, accuracy_iter: 0.116000, invalid_rate: 11.386000
I0510 12:25:55.981814 124672498300928 logging_util.py:133] [1251] eval_loss=0.69204, eval_accuracy=0.001, eval_accuracy_iter=0.00116, eval_invalid_rate=0.11386, eval_invalid_iter_rate=0.00036, ep=0.99909

### Auto decision
No automatic kill/launch in this loop. External MONITOR.py owns preemption resume; if DeiT windows < 8 or real code errors appear, next human/agent loop should inspect logs and decide.

## 2026-05-10 13:13:14 UTC
deit_windows=8
6380	sanity-run: full alignment with reference (WD mask, b2=0.999, LR schedule, cls init, stochastic depth schedule, final LN bias) — target 81.8	v6e-8-spot-gzy-favaxa	[1;32mRunning[0m (ep=151.22)
6381	phase2 Run H: attention head + aux CE loss (lambda=0.1). Tests whether CE supervision on backbone improves convergence speed vs pure diffusi	v6e-8-spot-gzy-qxxa8y	[1;32mRunning[0m (ep=154)
6382	phase2 Run E: zero-init out_proj in diffusion head; otherwise same as Run C (uniform schedule, biases+LS backbone). Tests whether zero-init 	v6e-8-spot-gzy-axuxm0	[1;32mRunning[0m (ep=151)
6383	phase2 Run G: larger diffusion head (512-dim, 4 layers, 8 heads) vs Run C (256-dim, 2 layers). Ablates head capacity: does more capacity hel	v6e-8-spot-gzy-06q7u9	[1;32mRunning[0m (ep=150)
6388	phase2 Run I: warm-start diffusion head from Phase 1 Run 3 (73.14% CE backbone). Tests whether pretrained backbone accelerates convergence v	v6e-8-spot-gzy-j3rqvs	[1;32mRunning[0m (ep=146)
6390	deit phase2 Run M: tiny-bit MLP (bit_dim=8, hidden=3072) + K10 multi-mask	v6e-8-spot-gzy-cz2ivo	[1;32mRunning[0m (ep=103)
6392	deit phase2 Run N: tiny-bit MLP + K10 + low-aug mild (no RA, reprob0, repeated1, wd0.02, sd0.05)	v6e-8-spot-gzy-3djlis	[1;32mRunning[0m (ep=128)
6405	deit phase2 Run L: cross-attention head over full patch tokens (4 layers) + K10 multi-mask	v6e-8-spot-gzy-z169mq	[1;32mRunning[0m (ep=8.15)

### Latest eval snapshot
### window=6380 tpu=v6e-8-spot-gzy-favaxa
tag=sanity-run: full alignment with reference (WD mask, b2=0.999, LR schedule, cls init, stochastic depth schedule, final LN bias) — target 81.8
status=[1;32mRunning[0m (ep=151.22)
logdir=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260509_212142_mxfpdz_kmh-tpuvm-v6e-8-spot-gzy-favaxa_asia-northeast1-b__b_lr_ep_eval
I0510 05:45:54.834356 138353402066944 logging_util.py:18] eval epoch: 79, loss: 1.957492, accuracy: 59.342000
I0510 05:45:54.834495 138353402066944 logging_util.py:133] [100080] eval_loss=1.9575, eval_accuracy=0.59342, ep=79.99
I0510 07:51:31.791923 138353402066944 logging_util.py:18] eval epoch: 99, loss: 1.774279, accuracy: 62.754000
I0510 07:51:31.792053 138353402066944 logging_util.py:133] [125100] eval_loss=1.7743, eval_accuracy=0.62754, ep=99.988
I0510 09:56:59.650715 138353402066944 logging_util.py:18] eval epoch: 119, loss: 1.834525, accuracy: 62.622000
I0510 09:56:59.650856 138353402066944 logging_util.py:133] [150120] eval_loss=1.8345, eval_accuracy=0.62622, ep=119.99
I0510 12:02:35.289051 138353402066944 logging_util.py:18] eval epoch: 139, loss: 1.834445, accuracy: 63.356000
I0510 12:02:35.289182 138353402066944 logging_util.py:133] [175140] eval_loss=1.8344, eval_accuracy=0.63356, ep=139.98

### window=6381 tpu=v6e-8-spot-gzy-qxxa8y
tag=phase2 Run H: attention head + aux CE loss (lambda=0.1). Tests whether CE supervision on backbone improves convergence speed vs pure diffusi
status=[1;32mRunning[0m (ep=154)
logdir=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260509_213033_is69hk_kmh-tpuvm-v6e-8-spot-gzy-qxxa8y_asia-northeast1-b__b_lr_ep_eval
I0510 07:42:26.887118 131896659294208 logging_util.py:18] eval epoch: 99, accuracy_iter: 53.334000, invalid_rate: 0.900000
I0510 07:42:26.887233 131896659294208 logging_util.py:133] [125100] eval_loss=0.45417, eval_accuracy=0.50792, eval_accuracy_iter=0.53334, eval_invalid_rate=0.009, eval_invalid_iter_rate=0.00022, ep=99.988
I0510 09:44:33.487574 131896659294208 logging_util.py:18] eval epoch: 119, loss: 0.437169, accuracy: 51.548000
I0510 09:44:33.487690 131896659294208 logging_util.py:18] eval epoch: 119, accuracy_iter: 53.886000, invalid_rate: 0.646000
I0510 09:44:33.487797 131896659294208 logging_util.py:133] [150120] eval_loss=0.43717, eval_accuracy=0.51548, eval_accuracy_iter=0.53886, eval_invalid_rate=0.00646, eval_invalid_iter_rate=0.00026, ep=119.99
I0510 11:46:41.752510 131896659294208 logging_util.py:18] eval epoch: 139, loss: 0.437972, accuracy: 52.506000
I0510 11:46:41.752627 131896659294208 logging_util.py:18] eval epoch: 139, accuracy_iter: 54.816000, invalid_rate: 0.378000
I0510 11:46:41.752727 131896659294208 logging_util.py:133] [175140] eval_loss=0.43797, eval_accuracy=0.52506, eval_accuracy_iter=0.54816, eval_invalid_rate=0.00378, eval_invalid_iter_rate=0.00016, ep=139.98

### window=6382 tpu=v6e-8-spot-gzy-axuxm0
tag=phase2 Run E: zero-init out_proj in diffusion head; otherwise same as Run C (uniform schedule, biases+LS backbone). Tests whether zero-init 
status=[1;32mRunning[0m (ep=151)
logdir=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260509_214004_nin5vh_kmh-tpuvm-v6e-8-spot-gzy-axuxm0_asia-northeast1-b__b_lr_ep_eval
I0510 07:55:13.494560 134033742792704 logging_util.py:18] eval epoch: 99, accuracy_iter: 1.016000, invalid_rate: 1.724000
I0510 07:55:13.494697 134033742792704 logging_util.py:133] [125100] eval_loss=0.67832, eval_accuracy=0.00426, eval_accuracy_iter=0.01016, eval_invalid_rate=0.01724, eval_invalid_iter_rate=0.00042, ep=99.988
I0510 09:57:22.967946 134033742792704 logging_util.py:18] eval epoch: 119, loss: 0.669255, accuracy: 0.806000
I0510 09:57:22.968081 134033742792704 logging_util.py:18] eval epoch: 119, accuracy_iter: 1.874000, invalid_rate: 2.144000
I0510 09:57:22.968196 134033742792704 logging_util.py:133] [150120] eval_loss=0.66925, eval_accuracy=0.00806, eval_accuracy_iter=0.01874, eval_invalid_rate=0.02144, eval_invalid_iter_rate=6e-05, ep=119.99
I0510 11:59:48.943911 134033742792704 logging_util.py:18] eval epoch: 139, loss: 0.659177, accuracy: 1.326000
I0510 11:59:48.944211 134033742792704 logging_util.py:18] eval epoch: 139, accuracy_iter: 3.172000, invalid_rate: 3.734000
I0510 11:59:48.944433 134033742792704 logging_util.py:133] [175140] eval_loss=0.65918, eval_accuracy=0.01326, eval_accuracy_iter=0.03172, eval_invalid_rate=0.03734, eval_invalid_iter_rate=0.00022, ep=139.98

### window=6383 tpu=v6e-8-spot-gzy-06q7u9
tag=phase2 Run G: larger diffusion head (512-dim, 4 layers, 8 heads) vs Run C (256-dim, 2 layers). Ablates head capacity: does more capacity hel
status=[1;32mRunning[0m (ep=150)
logdir=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260509_214224_qa2gek_kmh-tpuvm-v6e-8-spot-gzy-06q7u9_asia-northeast1-b__b_lr_ep_eval
I0510 08:00:28.689179 126804566546432 logging_util.py:18] eval epoch: 99, accuracy_iter: 2.658000, invalid_rate: 1.550000
I0510 08:00:28.689292 126804566546432 logging_util.py:133] [125100] eval_loss=0.66598, eval_accuracy=0.01044, eval_accuracy_iter=0.02658, eval_invalid_rate=0.0155, eval_invalid_iter_rate=4e-05, ep=99.988
I0510 10:04:26.730893 126804566546432 logging_util.py:18] eval epoch: 119, loss: 0.657741, accuracy: 1.850000
I0510 10:04:26.731019 126804566546432 logging_util.py:18] eval epoch: 119, accuracy_iter: 4.320000, invalid_rate: 3.918000
I0510 10:04:26.731139 126804566546432 logging_util.py:133] [150120] eval_loss=0.65774, eval_accuracy=0.0185, eval_accuracy_iter=0.0432, eval_invalid_rate=0.03918, eval_invalid_iter_rate=2e-05, ep=119.99
I0510 12:08:27.774757 126804566546432 logging_util.py:18] eval epoch: 139, loss: 0.644498, accuracy: 2.952000
I0510 12:08:27.774875 126804566546432 logging_util.py:18] eval epoch: 139, accuracy_iter: 6.636000, invalid_rate: 2.694000
I0510 12:08:27.774989 126804566546432 logging_util.py:133] [175140] eval_loss=0.6445, eval_accuracy=0.02952, eval_accuracy_iter=0.06636, eval_invalid_rate=0.02694, eval_invalid_iter_rate=0, ep=139.98

### window=6388 tpu=v6e-8-spot-gzy-j3rqvs
tag=phase2 Run I: warm-start diffusion head from Phase 1 Run 3 (73.14% CE backbone). Tests whether pretrained backbone accelerates convergence v
status=[1;32mRunning[0m (ep=146)
logdir=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260509_221135_gmkq7q_kmh-tpuvm-v6e-8-spot-gzy-j3rqvs_asia-northeast1-b__b_lr_ep_eval
I0510 08:25:42.097741 125830992791552 logging_util.py:18] eval epoch: 99, accuracy_iter: 59.288000, invalid_rate: 1.226000
I0510 08:25:42.097844 125830992791552 logging_util.py:133] [125100] eval_loss=0.34528, eval_accuracy=0.55438, eval_accuracy_iter=0.59288, eval_invalid_rate=0.01226, eval_invalid_iter_rate=0.0002, ep=99.988
I0510 10:27:42.504707 125830992791552 logging_util.py:18] eval epoch: 119, loss: 0.343700, accuracy: 57.410000
I0510 10:27:42.504832 125830992791552 logging_util.py:18] eval epoch: 119, accuracy_iter: 60.872000, invalid_rate: 0.686000
I0510 10:27:42.504936 125830992791552 logging_util.py:133] [150120] eval_loss=0.3437, eval_accuracy=0.5741, eval_accuracy_iter=0.60872, eval_invalid_rate=0.00686, eval_invalid_iter_rate=0.00022, ep=119.99
I0510 12:29:41.421755 125830992791552 logging_util.py:18] eval epoch: 139, loss: 0.334276, accuracy: 58.986000
I0510 12:29:41.421875 125830992791552 logging_util.py:18] eval epoch: 139, accuracy_iter: 62.470000, invalid_rate: 0.444000
I0510 12:29:41.421976 125830992791552 logging_util.py:133] [175140] eval_loss=0.33428, eval_accuracy=0.58986, eval_accuracy_iter=0.6247, eval_invalid_rate=0.00444, eval_invalid_iter_rate=0.00024, ep=139.98

### window=6390 tpu=v6e-8-spot-gzy-cz2ivo
tag=deit phase2 Run M: tiny-bit MLP (bit_dim=8, hidden=3072) + K10 multi-mask
status=[1;32mRunning[0m (ep=103)
logdir=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_024012_fw1rbv_kmh-tpuvm-v6e-8-spot-gzy-cz2ivo_us-east5-b__b_lr_ep_eval
I0510 08:47:48.584914 130698839156736 logging_util.py:18] eval epoch: 59, accuracy_iter: 54.348000, invalid_rate: 0.744000
I0510 08:47:48.585018 130698839156736 logging_util.py:133] [75060] eval_loss=0.39957, eval_accuracy=0.48126, eval_accuracy_iter=0.54348, eval_invalid_rate=0.00744, eval_invalid_iter_rate=0.00016, ep=59.993
I0510 10:49:41.939933 130698839156736 logging_util.py:18] eval epoch: 79, loss: 0.411026, accuracy: 51.186000
I0510 10:49:41.940230 130698839156736 logging_util.py:18] eval epoch: 79, accuracy_iter: 56.400000, invalid_rate: 0.660000
I0510 10:49:41.940362 130698839156736 logging_util.py:133] [100080] eval_loss=0.41103, eval_accuracy=0.51186, eval_accuracy_iter=0.564, eval_invalid_rate=0.0066, eval_invalid_iter_rate=0.00028, ep=79.99
I0510 12:51:36.235283 130698839156736 logging_util.py:18] eval epoch: 99, loss: 0.430625, accuracy: 52.132000
I0510 12:51:36.235399 130698839156736 logging_util.py:18] eval epoch: 99, accuracy_iter: 56.992000, invalid_rate: 0.660000
I0510 12:51:36.235523 130698839156736 logging_util.py:133] [125100] eval_loss=0.43062, eval_accuracy=0.52132, eval_accuracy_iter=0.56992, eval_invalid_rate=0.0066, eval_invalid_iter_rate=0.00018, ep=99.988

### window=6392 tpu=v6e-8-spot-gzy-3djlis
tag=deit phase2 Run N: tiny-bit MLP + K10 + low-aug mild (no RA, reprob0, repeated1, wd0.02, sd0.05)
status=[1;32mRunning[0m (ep=128)
logdir=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_024337_vt4mt2_kmh-tpuvm-v6e-8-spot-gzy-3djlis_asia-northeast1-b__b_lr_ep_eval
I0510 09:16:57.273761 126447346067456 logging_util.py:18] eval epoch: 79, accuracy_iter: 60.954000, invalid_rate: 0.696000
I0510 09:16:57.273892 126447346067456 logging_util.py:133] [100080] eval_loss=0.39084, eval_accuracy=0.56896, eval_accuracy_iter=0.60954, eval_invalid_rate=0.00696, eval_invalid_iter_rate=4e-05, ep=79.99
I0510 10:54:51.424077 126447346067456 logging_util.py:18] eval epoch: 99, loss: 0.414502, accuracy: 57.566000
I0510 10:54:51.424340 126447346067456 logging_util.py:18] eval epoch: 99, accuracy_iter: 61.624000, invalid_rate: 0.826000
I0510 10:54:51.424509 126447346067456 logging_util.py:133] [125100] eval_loss=0.4145, eval_accuracy=0.57566, eval_accuracy_iter=0.61624, eval_invalid_rate=0.00826, eval_invalid_iter_rate=0.00018, ep=99.988
I0510 12:33:04.162859 126447346067456 logging_util.py:18] eval epoch: 119, loss: 0.420070, accuracy: 59.060000
I0510 12:33:04.162986 126447346067456 logging_util.py:18] eval epoch: 119, accuracy_iter: 62.800000, invalid_rate: 0.722000
I0510 12:33:04.163091 126447346067456 logging_util.py:133] [150120] eval_loss=0.42007, eval_accuracy=0.5906, eval_accuracy_iter=0.628, eval_invalid_rate=0.00722, eval_invalid_iter_rate=0.0001, ep=119.99

### window=6405 tpu=v6e-8-spot-gzy-z169mq
tag=deit phase2 Run L: cross-attention head over full patch tokens (4 layers) + K10 multi-mask
status=[1;32mRunning[0m (ep=8.15)
logdir=/kmh-nfs-ssd-us-mount/logs/sqa/paligemma-baseline/20260510_121100_lkmvun_kmh-tpuvm-v6e-8-spot-gzy-z169mq_asia-northeast1-b__b_lr_ep_eval
I0510 12:25:55.981643 124672498300928 logging_util.py:18] eval epoch: 0, loss: 0.692041, accuracy: 0.100000
I0510 12:25:55.981736 124672498300928 logging_util.py:18] eval epoch: 0, accuracy_iter: 0.116000, invalid_rate: 11.386000
I0510 12:25:55.981814 124672498300928 logging_util.py:133] [1251] eval_loss=0.69204, eval_accuracy=0.001, eval_accuracy_iter=0.00116, eval_invalid_rate=0.11386, eval_invalid_iter_rate=0.00036, ep=0.99909

### Auto decision
No automatic kill/launch in this loop. External MONITOR.py owns preemption resume; if DeiT windows < 8 or real code errors appear, next human/agent loop should inspect logs and decide.

