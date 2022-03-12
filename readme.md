# TQA Training Library

This module contains the common functions and types to be used by other portions of the TweetQA system, such as the web API and the pipeline. It is also capable of training runs manually to be used with local hardware.

## How to use

### Install Dependency

The library is easy to reference. Use the following pip command to install the module where x.y.z is the tag you want to install, or a branch name:

```bash
pip install git+https://github.com/sweng480-team23/tqa-training-lib@x.y.z
```

For example, to install the master branch, use:

```bash
pip install git+https://github.com/sweng480-team23/tqa-training-lib@main
```

To install version 1.2.3, use:

```bash
pip install git+https://github.com/sweng480-team23/tqa-training-lib@1.2.3
```

### Usage

From there, you can simply reference the parts of the library module that you need like so:

```py
from tqa_training_lib.trainers.tf_tweetqa_trainer import TFTweetQATrainer
```

## IMPORTANT

If you experience \[CLS\] issues with Tensorflow, try running the training with `$env:TF_GPU_ALLOCATOR="cuda_malloc_async"` (powershell). Example for manual training:

```powershell
$env:TF_GPU_ALLOCATOR="cuda_malloc_async"; python .\tf_train.py
```

## Todo

- Improve documentation
- Probably add tests
- Merge manual training files (tf_train and torch_train) into a single file, command line args to override training params
- Perform validation on val_dataset
- ModelRunner check for valid model file at location and throw error if missing
- Fix TF training issue for > ~ 580 training points (try CPU? seems to be memory related)
- Investigate problem for low scores on torch models (try w/o 🤗 trainer?)
- Fix install_requires in setup.py (see comments in file)
