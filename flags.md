when we print FLAGS:

```bash
flags:  
main.py:
  --config: File path to the training hyperparameter configuration.
    (default: 'None')
  --[no]debug: Debugging mode.
    (default: 'false')
  --workdir: Directory to store model data.

absl.app:
  -?,--[no]help: show this help
    (default: 'false')
  --[no]helpfull: show full help
    (default: 'false')
  --[no]helpshort: show this help
    (default: 'false')
  --[no]helpxml: like --helpfull, but generates XML output
    (default: 'false')
  --[no]only_check_args: Set to true to validate args and exit.
    (default: 'false')
  --[no]pdb: Alias for --pdb_post_mortem.
    (default: 'false')
  --[no]pdb_post_mortem: Set to true to handle uncaught exceptions with PDB post mortem.
    (default: 'false')
  --profile_file: Dump profile information to a file (for python -m pstats). Implies --run_with_profiling.
  --[no]run_with_pdb: Set to true for PDB debug mode
    (default: 'false')
  --[no]run_with_profiling: Set to true for profiling the script. Execution will be slower, and the output format might change over time.
    (default: 'false')
  --[no]use_cprofile_for_profiling: Use cProfile instead of the profile module for profiling. This has no effect unless --run_with_profiling is set.
    (default: 'true')

absl.logging:
  --[no]alsologtostderr: also log to stderr?
    (default: 'false')
  --log_dir: directory to write logfiles into
    (default: '')
  --logger_levels: Specify log level of loggers. The format is a CSV list of `name:level`. Where `name` is the logger name used with `logging.getLogger()`, and
    `level` is a level name  (INFO, DEBUG, etc). e.g. `myapp.foo:INFO,other.logger:DEBUG`
    (default: '')
  --[no]logtostderr: Should only log to stderr?
    (default: 'false')
  --[no]showprefixforinfo: If False, do not prepend prefix to info messages when it's logged to stderr, --verbosity is set to INFO level, and python logging is
    used.
    (default: 'true')
  --stderrthreshold: log messages at this level, or more severe, to stderr in addition to the logfile.  Possible values are 'debug', 'info', 'warning', 'error',
    and 'fatal'.  Obsoletes --alsologtostderr. Using --alsologtostderr cancels the effect of this flag. Please also note that this flag is subject to
    --verbosity and requires logfile not be stderr.
    (default: 'fatal')
  -v,--verbosity: Logging verbosity level. Messages logged at this level or lower will be included. Set to 1 for debug logging. If the flag was not set or
    supplied, the value will be changed from the default of -1 (warning) to 0 (info) after flags are parsed.
    (default: '-1')
    (an integer)

absl.testing.absltest:
  --test_random_seed: Random seed for testing. Some test frameworks may change the default value of this flag between runs, so it is not appropriate for seeding
    probabilistic tests.
    (default: '301')
    (an integer)
  --test_randomize_ordering_seed: If positive, use this as a seed to randomize the execution order for test cases. If "random", pick a random seed to use. If 0
    or not set, do not randomize test case execution order. This flag also overrides the TEST_RANDOMIZE_ORDERING_SEED environment variable.
    (default: '')
  --test_srcdir: Root of directory tree where source files live
    (default: '')
  --test_tmpdir: Directory for temporary testing files
    (default: '/tmp/absl_testing')
  --xml_output_file: File to store XML test results
    (default: '')

chex._src.fake:
  --[no]chex_assert_multiple_cpu_devices: Whether to fail if a number of CPU devices is less than 2.
    (default: 'false')
  --chex_n_cpu_devices: Number of CPU threads to use as devices in tests.
    (default: '1')
    (an integer)

chex._src.variants:
  --[no]chex_skip_pmap_variant_if_single_device: Whether to skip pmap variant if only one device is available.
    (default: 'true')

cloud_tpu_client.client:
  --[no]runtime_oom_exit: Exit the script when the TPU runtime is OOM.
    (default: 'true')

ml_collections.config_flags.config_flags:
  --config.batch_size: An override of config's field batch_size
    (default: '256')
    (an integer)
  --config.dataset.num_workers: An override of config's field dataset.num_workers
    (default: '4')
    (an integer)
  --config.dataset.prefetch_factor: An override of config's field dataset.prefetch_factor
    (default: '2')
    (an integer)
  --config.dataset.root: An override of config's field dataset.root
    (default: '/kmh-nfs-us-mount/data/imagenet')
  --config.learning_rate: An override of config's field learning_rate
    (default: '0.1')
    (a number)
  --config.log_per_step: An override of config's field log_per_step
    (default: '100')
    (an integer)
  --config.model: An override of config's field model
    (default: 'ResNet50')
  --config.num_epochs: An override of config's field num_epochs
    (default: '5')
    (an integer)

tensorflow.python.ops.parallel_for.pfor:
  --[no]op_conversion_fallback_to_while_loop: DEPRECATED: Flag is ignored.
    (default: 'true')

tensorflow.python.tpu.tensor_tracer_flags:
  --delta_threshold: Log if history based diff crosses this threshold.
    (default: '0.5')
    (a number)
  --[no]tt_check_filter: Terminate early to check op name filtering.
    (default: 'false')
  --[no]tt_single_core_summaries: Report single core metric and avoid aggregation.
    (default: 'false')

absl.flags:
  --flagfile: Insert flag definitions from the given file into the command line.
    (default: '')
  --undefok: comma-separated list of flag names that it is okay to specify on the command line even if the program does not define a flag with that name.
    IMPORTANT: flags in this list that have arguments MUST use the --flag=value format.
    (default: '')
```

when we print FLAGS.config:

```bash
FLAGS.config:  batch_size: 1024
checkpoint_per_epoch: 20
dataset:
  cache: false
  name: imagenet
  num_workers: 32
  pin_memory: false
  prefetch_factor: 2
  root: ./imagenet_fake
eval_per_epoch: 1
half_precision: true
learning_rate: 0.1
log_per_epoch: -1
log_per_step: 20
model: ViT_debug
momentum: 0.9
num_epochs: 100
num_train_steps: 1
prefetch: 10
seed: 0
shuffle_buffer_size: 2048
steps_per_eval: 1
warmup_epochs: 5
```