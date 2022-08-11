# PipeEdge

PipeEdge is an inference framework that pipelines neural network (e.g., transformer) model shards on distributed devices.
It includes an automatic partition scheduler which maps model layers to devices to optimize throughput.


## Prerequisites

System dependencies:

* Python >= 3.7
* Compiler with C++17 support
* CMake >= 3.8 (for C++17 support)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp) >= 0.6.0

On MacOS:

```sh
brew install cmake yaml-cpp
```

On Debian (>= buster) or Debian-based Linux (including Ubuntu >= 20.04):

```sh
sudo apt-get install build-essential cmake libyaml-cpp-dev
```

We recommend using a Python virtual environment (`virtualenv`), e.g., on Debian-based Linux:

```sh
sudo apt-get install python3-venv
```

or directly with a system-installed `pip`:

```sh
pip3 install virtualenv
```

Create and activate the virtualenv:

```sh
python3 -m venv .venv
. .venv/bin/activate
```

Install the development package, Python package dependencies, and runtime application dependencies with:

```sh
pip install -U pip
pip install -e .[runtime]
```

Download model weight files (ViT files are from [Google Cloud](https://console.cloud.google.com/storage/browser/vit_models)):

```sh
python save_model_weights.py
```

### Optional dependencies:

System dependencies required for runtime monitoring:

* [EnergyMon](https://github.com/energymon/energymon) - with a system-appropriate "default" library (which may have transitive dependencies)


### ImageNet Dataset

The ILSVRC 2012 dataset (a.k.a. ImageNet-1K) cannot be downloaded automatically since access requires registration.
Register at [image-net.org](https://image-net.org/download-images.php), then download the training (`ILSVRC2012_img_train.tar`: \~138 GB) and validation (`ILSVRC2012_img_val.tar`: \~6.3 GB) image archives along with the devkit archive (`ILSVRC2012_devkit_t12.tar.gz`) and place them in a common directory.

The archives must be preprocessed to create a usable directory structure.
If you placed the archives in a directory called `ImageNet`:

```sh
python tools/imagenet_preprocess.py ImageNet
```

The script may take several minutes or more to run, depending on the storage disk speed.
Alternatively, there are Bash script examples online for achieving the same result.


## Usage

For full usage help, run:

```sh
python runtime.py -h
```

To run with default parameters (using ViT-Base) on a single node:

```sh
python runtime.py 0 1
```

To run on multiple nodes, e.g., with 2 stages and even partitioning, on rank 0:

```sh
python runtime.py 0 2 -pt 1,24,25,48
```

and on rank 1:

```sh
python runtime.py 1 2 -pt 1,24,25,48
```

### Partitioning

For example, the ViT-Base model has 12 layers, so the range is [1, 12*4] = [1, 48].

An even partitioning for 2 nodes is:
```
partition = [1,24,25,48]
```

An uneven partitioning for 2 nodes could be:
```
partition = [1,47,48,48]
```

A partitioning for 4 nodes could be:
```
partition = [1,4,5,8,9,20,21,48]
```


## Automatic Partition Scheduling

In summary, the `sched-pipeline` scheduling application uses three input YAML files to map model partitions to devices (hosts).
Automated profiling helps produce two of these files; the third lists available hosts and is straightforward to create for your deployment environment.
For detailed instructions and documentation, see [README_Profiler.md](README_Profiler.md) and [README_Scheduler.md](README_Scheduler.md).

Point `runtime.py` to the YAML files using the options `-sm/--sched-models-file`, `--sdt/--sched-dev-types-file`, and `-sd/--sched-dev-file`.
The runtime passes these through to the previously compiled scheduler application, along with other configurations like the model name and microbatch size.
Then map the hosts specified in the third YAML file to the distributed ranks in your runtime using the `-H/--hosts` option.
Do not specify the `-pt/--partition` option, which is for manually specifying the schedule and takes precedence over automated scheduling.
