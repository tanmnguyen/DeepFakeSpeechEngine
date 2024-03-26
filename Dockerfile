# Use the official Ubuntu 20.04 base image
FROM ubuntu:20.04

# Update package lists and install essential packages
RUN apt-get update && \
    apt-get install -y \
    sudo \
    curl \
    wget \
    gnupg \
    lsb-release

# Set the environment variables
ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app 

# Install python3.9
RUN apt-get update && apt-get install -y python3.9 python3.9-dev python3.9-distutils
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --config python3
RUN apt-get update && apt-get install -y rsync

RUN apt-get -y install build-essential

# Install Kaldi dependencies
RUN apt-get install -y unzip sox gfortran python2.7 automake autoconf git libtool subversion

RUN git clone https://github.com/kaldi-asr/kaldi.git
RUN touch kaldi/helloworld.txt
RUN kaldi/tools/extras/install_mkl.sh
RUN cd kaldi/tools && make -j $(nproc)
RUN cd kaldi/src && ./configure --shared && make depend -j $(nproc) && make -j $(nproc)

RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
RUN apt-get install -y libxml2
RUN chmod +x cuda_11.8.0_520.61.05_linux.run

RUN ./cuda_11.8.0_520.61.05_linux.run \
  --silent \
  --toolkit \
  --installpath=/star-fj/fangjun/software/cuda-11.8.0 \
  --no-opengl-libs \
  --no-drm \
  --no-man-page

RUN wget https://huggingface.co/csukuangfj/cudnn/resolve/main/cudnn-linux-x86_64-8.9.1.23_cuda11-archive.tar.xz
RUN tar xvf cudnn-linux-x86_64-8.9.1.23_cuda11-archive.tar.xz --strip-components=1 -C /star-fj/fangjun/software/cuda-11.8.0

ENV CUDA_HOME=/star-fj/fangjun/software/cuda-11.8.0
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

ENV CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
ENV CUDA_TOOLKIT_ROOT=$CUDA_HOME
ENV CUDA_BIN_PATH=$CUDA_HOME
ENV CUDA_PATH=$CUDA_HOME
ENV CUDA_INC_PATH=$CUDA_HOME/targets/x86_64-linux
ENV CFLAGS=-I$CUDA_HOME/targets/x86_64-linux/include:$CFLAGS

# install pip 
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

RUN pip3 install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install k2==1.24.3.dev20230719+cpu.torch2.0.1 -f https://k2-fsa.github.io/k2/cpu.html
RUN pip install git+https://github.com/lhotse-speech/lhotse

RUN git clone https://github.com/k2-fsa/icefall && cd icefall/ && pip install -r ./requirements.txt 
ENV PYTHONPATH=/app/icefall:$PYTHONPATH

# install Kadifeat Linux Cuda 
RUN pip install torch==2.1.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install kaldifeat==1.25.3.dev20231221+cuda12.1.torch2.1.2 -f https://csukuangfj.github.io/kaldifeat/cuda.html

RUN apt install python-is-python3
# # install wenet 
# RUN pip install git+https://github.com/wenet-e2e/wenet.git

# # install voicebox 
# RUN git clone https://github.com/voiceboxneurips/voicebox.git
# RUN  cd voicebox && pip install -r requirements.txt && pip install -e . &&  chmod -R u+x scripts/