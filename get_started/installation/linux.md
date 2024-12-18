# Installing on Linux

pysensing can be installed and used on various Linux distributions. Depending on your system and compute requirements, your experience with pysensing on Linux may vary in terms of processing time. It is recommended, but not required, that your Linux system has an NVIDIA GPU in order to harness the full power of pysensing's [CUDA](https://developer.nvidia.com/cuda-zone) support.
<!-- [support](https://pysensing.org/tutorials/beginner/blitz/tensor_tutorial.html?highlight=cuda#cuda-tensors). -->

## Prerequisites
<!-- {: #linux-prerequisites}

### Supported Linux Distributions

pysensing is supported on Linux distributions that use [glibc](https://www.gnu.org/software/libc/) >= v2.17, which include the following:

* [Arch Linux](https://www.archlinux.org/download/), minimum version 2012-07-15
* [CentOS](https://www.centos.org/download/), minimum version 7.3-1611
* [Debian](https://www.debian.org/distrib/), minimum version 8.0
* [Fedora](https://getfedora.org/), minimum version 24
* [Mint](https://linuxmint.com/download.php), minimum version 14
* [OpenSUSE](https://software.opensuse.org/), minimum version 42.1
* [PCLinuxOS](https://www.pclinuxos.com/get-pclinuxos/), minimum version 2014.7
* [Slackware](http://www.slackware.com/getslack/), minimum version 14.2
* [Ubuntu](https://www.ubuntu.com/download/desktop), minimum version 13.04

> The install instructions here will generally apply to all supported Linux distributions. An example difference is that your distribution may support `yum` instead of `apt`. The specific examples shown were run on an Ubuntu 18.04 machine. -->

### Python


Python 3.7 or greater is generally installed by default on any of our supported Linux distributions, which meets our requirments.

We recommend starting with Anaconda and pip to create a controlled environment for your pysensing installation.


#### Install Anaconda

Download and install Anaconda using the [command-line installer](https://www.anaconda.com/download/#linux).

```bash
# The version of Anaconda may be different depending on when you are installing`
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
# and follow the prompts. The defaults are generally good.`
```

> You may have to open a new terminal or re-source your `~/.bashrc `to get access to the `conda` command. 

Create a pysesning enviroment with specified python version, we recommend to use python >= 3.10:
```bash
conda create -n pysensing python=3.10 
conda activate pysensing
```
#### PyTorch
Follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install PyTorch based on your system’s capabilities.

- **For systems with CUDA support** (replace `cu118` with your CUDA version):
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- **For systems without CUDA support** :
  ```bash
  pip install torch torchvision torchaudio
  ```
## Installation

<!-- ### Anaconda
{: #linux-anaconda}

#### No CUDA

To install pysensing via Anaconda, and do not have a [CUDA-capable](https://developer.nvidia.com/cuda-zone) system or do not require CUDA, in the above selector, choose OS: Linux, Package: Conda and CUDA: None.
Then, run the command that is presented to you.

#### With CUDA

To install pysensing via Anaconda, and you do have a [CUDA-capable](https://developer.nvidia.com/cuda-zone) system, in the above selector, choose OS: Linux, Package: Conda and the CUDA version suited to your machine. Often, the latest CUDA version is better.
Then, run the command that is presented to you. -->


### pip
The pysensing package itself can be installed from pypi by:
```bash
pip install pysensing
```
### Building from source
If you want to build from source and get and latest version:
```bash
pip install --upgrade git+https://github.com/pysensing/pysensing.git
```


## Verification

To ensure that `pysensing` was installed correctly, verify the installation by running a sample tutorial notebook.

1. Clone the `pysensing` repository:
```bash
git clone https://github.com/pysensing/pysensing.git
```

2. Install Jupyter via pip:
```bash
pip install jupyter
```

3. Navigate to the tutorial directory and launch Jupyter Notebook:
```bash
cd pysensing/pysensing/acoustic/tutorials/
jupyter notebook
```

This will open the tutorial notebooks in your default web browser.