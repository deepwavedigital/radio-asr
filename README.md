# Radio-ASR
This package is used to convert demodulated audio signals into text. It leverages the
NeMo ASR model.

**NOTE: The installation order matters here. Follow the instructions step-by-step**

## Installation For JetPack 4.6 (AirStack 0.5+)
Some of the pip builds for NLP pkgs need a C++ compiler. Building some of NeMo's ASR
dependencies also requires a Rust compiler


### Prerequisites

* Install `git-lfs` in order to access the large binary files in this repository

  ```
  sudo apt-get install git-lfs
  git lfs install
  ```

* Install Binary Dependencies of NVIDIA-built PyTorch package
  ```
  sudo apt-get install libopenblas-base libopenmpi-dev
  ```

### Create Initial Environment
Note: Always add compilers when creating the initial environment: they have activation
scripts to set the environment, so otherwise would have to deactivate and reactivate the
environment.

```
conda create -n gnuradio-asr -c file:///opt/deepwave/conda-channels/airstack-conda python=3.6 compilers rust numpy scipy matplotlib mamba
conda activate gnuradio-asr
mamba update mamba
mamba install -c file:///opt/deepwave/conda-channels/airstack-conda gnuradio soapysdr-module-airt
mamba install scikit-learn onnx ipython pandas notebook numba click=7 cython h5py \
              sympy editdistance nltk grpcio markdown werkzeug tensorboard=2.4
```

### Install PyTorch
See documentation [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-9-0-now-available/72048)
for installing PyTorch on JetPack.

* The steps for PyTorch 1.9.0 (downloading the package from NVidia)
  ```
  wget https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl
  mv h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl torch-1.9.0-cp36-cp36m-linux_aarch64.whl
  pip install torch-1.9.0-cp36-cp36m-linux_aarch64.whl
  ```

  Note that the torch wheel is saved to this repository just in case it stops being published.
  You can alternately just run:

  ```
  pip install pkgs/torch-1.9.0-cp36-cp36m-linux_aarch64.whl
  ```

### Install NeMo
* Clone the repository
  ```
  git clone https://github.com/NVIDIA/NeMo
  cd NeMo
  git checkout v1.3.0
  ```

* Basic requirements for NeMo
  ```
  cd NeMo/requirements
  pip install -r requirements.txt
  pip install -r requirements_asr.txt
  ```

* The model depends on a c++ package that's in conda, but not built for linux-aarch64. We
  could clone the feedstock and build it ourselves, but it also doesn't build cleanly.
  It's for Japanese language support which we don't need right now so we remove it.
  ```
  cd NeMo
  patch -p1 < ../nemo-1.3.patch
  ```

* The NLP code installs another package that has a C++ dependency that we don't have, so
  replace it with the pure python version of the same package here...
  ```
  pip uninstall opencc
  pip install opencc-python-reimplemented
  ```

* Add NeMo to our environment
  ```
  pip install ./NeMo
  ```

* One last step: build/install the external libraries necessary to run the beam search
  decoders & language models.
  ```
  cd NeMo
  scripts/asr_language_modeling/ngram_lm/install_beamsearch_decoders.sh
  ```

### Install Radio-ASR

Run this command from the folder containing this readme (i.e. the top of this git repo)
```
pip install -e .
```

Now you SHOULD be done and can run the transcription examples. Whew!
