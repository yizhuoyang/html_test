# Installing on Windows
{:.no_toc}

pysensing can be installed and used on various Windows distributions. Depending on your system and compute requirements, your experience with pysensing on Windows may vary in terms of processing time. It is recommended, but not required, that your Windows system has an NVIDIA GPU in order to harness the full power of pysensing's [CUDA](https://developer.nvidia.com/cuda-zone) [support](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html?highlight=cuda#cuda-tensors).

## Prerequisites
{: #windows-prerequisites}

### Supported Windows Distributions

pysensing is supported on the following Windows distributions:

* [Windows 10](https://www.microsoft.com/en-us/software-download/windows10ISO) or greater recommended.

> The install instructions here will generally apply to all supported Windows distributions. The specific examples shown will be run on a Windows 11 Home machine

### Python
{: #windows-python}

Currently, pysensing on Windows only supports Python 3.7 or greater; Python 2.x is not supported.

As it is not installed by default on Windows, there are multiple ways to install Python:

* [Chocolatey](https://chocolatey.org/)
* [Python website](https://www.python.org/downloads/windows/)
* [Anaconda](https://www.anaconda.com/download/#windows)

> If you decide to use Chocolatey, and haven't installed Chocolatey yet, ensure that you are [running your command prompt as an administrator](https://www.howtogeek.com/194041/how-to-open-the-command-prompt-as-administrator-in-windows-8.1/).

For a Chocolatey-based install, run the following command in an [administrative command prompt](https://www.howtogeek.com/194041/how-to-open-the-command-prompt-as-administrator-in-windows-8.1/):

```bash
choco install python
```

### Package Manager
{: #windows-package-manager}

To install the pysensing binaries, you will need to use at least one of two supported package managers: [Anaconda](https://www.anaconda.com/download/#windows) and [pip](https://pypi.org/project/pip/). 

#### pip

If you installed Python by any of the recommended ways [above](#windows-python), [pip](https://pypi.org/project/pip/) will have already been installed for you.

#### Anaconda

Follow the instructions in [Anaconda Download](https://www.anaconda.com/download/#windows), and you will install the Anaconda including the Anaconda Navigator, which is intuitive for package management and recommended for Windows users.

> Make sure you have added your `conda-directory` and `conda-directory/Scripts` to your `PATH` vairables after Anaconda installation. Otherwise, the `conda` command in Windows command-line tool is not recognized. 

### CUDA

Before installing PyTorch, it will be better to check the compatibility of your device. Please refer to the [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver) and [NVIDIA support matrix](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html).

Two softwares need to be downloaded from NVIDIA for CUDA utility: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn-downloads).

> Note that `CUDA Toolkit` should be installted first, which is the basic of `cuDNN`.

#### CUDA Toolkit

Please follow the instructions in [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and run the executable for installation. 

> After installation, go to the Windows `system variabels` to check two variables: `CUDA_PATH` and `CUDA_PATH_V12_1` (12.1 version is just for example). If these two don't exist, please create them and set both their values to your `CUDA-installed-directory`. Then, open the command-line tool and input `nvcc -V` to check the success of installation.

#### cuDNN

When the CUDA Toolkit is ready, download the [cuDNN](https://developer.nvidia.com/cudnn-downloads) and compress the items. Then, copy the dirs `lib, bin, include` under the `cuDNN` compressed directory to the root directory of the `CUDA-installed-directory`. 

> Make sure you have added the following values to the `PATH` variables: `CUDA-installed-directory/bin`, `CUDA-installed-directory/libnvvp`, `CUDA-installed-directory/include` and `CUDA-installed-directory/lib`. 

### PyTorch

To avoid conflicts with other developments, it is recommended to create a virtual environment for PyTorch and pysensing installation.

For instance, to install PyTorch in a conda environment: 
- **Conda environment creation**:
  ```bash
  conda create -n your_dev python=3.10 
  conda activate your_dev
  ```
- **For systems with CUDA support** (replace `cu121` with your CUDA version):
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- **For systems without CUDA support** :
  ```bash
  pip install torch torchvision torchaudio
  ```

## Installation
{: #windows-installation}

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
{: #windows-verification}

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
