跑实验指南。

首先找卡：终端运行 tou (如果你不能运行，去~/.bashrc里面找对应的命令)
这个会输出所有可用的 TPU.

重要： 你只允许用 v5p <= 64 或者 v6e <= 32 的卡.

跑实验的方法：

找到卡之后，你需要找一个对应的 alias (需要根据卡的zone来确定。 具体很多这方面的逻辑你应该参考/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/MONITOR.py)

alias形如v5-64-tmp201这种

然后你运行 ftmd <ka> <alias> ; tpu run <ka> sqa dir=7

7 就是我设置的对这个目录的编号，你可以看 tpu ls sqa来查看。你可以用tpu set-cur <idx> sqa 来把当前目录设置，注意别乱覆盖。

上面这个命令会把卡放到这个alias上然后mount-disk (配环境)，然后跑任务。

跑任务的方式你不需要也不应该干预，这个是已经包装好的，不许乱动。跑上的任务可以用 tcs 查看，这里面有我的所有 job，你只需要关心 DeiT 相关的任务。

一个任务有对应的tmux window, 在sqa:<window_id> 这种。你可以看.bashrc里面有一个叫 cl 的命令，能access这个window的logfile，不过可能你不能直接使用这个接口因为它的原理是code ...， 在vscode中打开。不过总之你有接口查看logdir，类似你也可以找stagedir。

tcs 会显示任务状态。如果error你应该进去看一眼，如果是代码报错就fix一下。如果是卡被preempt了，你不需要做任何事情。

因为我有一个开着的自动resume任务的脚本，会定时检测preempt的job然后resume它。所以你不需要管，至少别改动resume的代码。你实在想看就看/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_manager/MONITOR.py。

最后等到你跑完任务，你应该先在 results.md 里面把结果，任务的 wandb link 都记录下来。然后你可以规划后面的job。

你可以同时跑多个job，并且一般推荐你这么做。空卡多的时候你就可以多跑几个。

超参不需要搜的太仔细，比如lr每次double就可以。搜太仔细没意义，我们remain high level一点。不过重要的design choice是需要搜的。

写新代码如果挺不一样的时候就开个新branch。或者你可以copy一份这个文件夹到外面去。

最后，你不应该修改这个文件夹下面的任何.sh文件，因为他们实际上是 soft link 到别的目录的。你改了别的就也都改了，所以你不允许改。