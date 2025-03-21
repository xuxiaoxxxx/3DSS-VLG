# Installation Instruction

Start by cloning the repo:
```bash
git clone https://github.com/xuxiaoxxxx/3DSS-VLG.git
cd 3DSS-VLG
```

You can create an anaconda environment called `3DSS-VLG` as below. For linux, you need to install `libopenexr-dev` before creating the environment.

```bash
sudo apt-get install libopenexr-dev # for linux
conda create -n vlg python=3.8
conda activate vlg
conda install openblas-devel -c anaconda
```

Step 1: install PyTorch(cuda 10.2 / cuda 11.1):

```bash
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Step 2: install MinkowskiNet:
```
pip install ninja
sudo apt install build-essential python3-dev libopenblas-dev
```
If you do not have sudo right, try the following:
```
pip install ninja
conda install openblas-devel -c anaconda
```

If your cuda version is 10.2, you should need to make sure your gcc and g++ versions are 7.5.0 and run that:
```
export CXX=g++-7
```

And now install MinkowskiNet, directly run this:
```
cd MinkowskiNet
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

Step 3: install all the remaining dependencies:
```bash
pip install -r requirements.txt
```