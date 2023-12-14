FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

ARG MMDEPLOY_VERSION=main
ENV NVIDIA_VISIBLE_DEVICE all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib/python3.8/dist-packages/opencv-python.libs${LD_LIBRARY_PATH}"
ENV TENSORRT_DIR="/usr/include/aarch64-linux-gnu"

ENV DEBIAN_FRONTEND=nointeractive
ENV FORCE_CUDA="1"

USER root
WORKDIR /root/workspace

# install dependencies
RUN  apt-get update &&\ 
     apt-get install -y vim wget libspdlog-dev libssl-dev libpng-dev pkg-config libhdf5-103 libhdf5-dev --no-install-recommends &&\
     python3 -m pip install onnx versioned-hdf5

# install onnxruntime
RUN wget https://nvidia.box.com/shared/static/mvdcltm9ewdy2d5nurkiqorofz1s53ww.whl -O onnxruntime_gpu-1.15.1.whl &&\
    python3 -m pip install --no-cache-dir onnxruntime_gpu-1.15.1-cp38-cp38-linux_aarch64.whl

# install mmcv
RUN git clone --branch 2.x https://github.com/open-mmlab/mmcv.git &&\
    python3 -m pip install --no-cache-dir opencv-python==4.5.4.60 opencv-contrib-python==4.5.4.60 opencv-python-headless==4.5.4.60 &&\
    MMCV_WITH_OPS=1 python3 -m pip install -e .

# build ppl.cv
RUN git clone https://github.com/openppl-public/ppl.cv.git &&\
    echo "export PPLCV_DIR=/root/workspace/ppl.cv" >> ~/.bashrc &&\
    ./build.sh cuda
  
# build mmdeploy
RUN git clone --recursive -b $MMDEPLOY_VERSION --depth 1 https://github.com/open-mmlab/mmdeploy &&\
    cd mmdeploy &&\
    mkdir -p build && cd build &&\
    cmake .. \
    	-DMMDEPLOY_TARGET_BACKENDS="trt" \
    	-DTENSORRT_DIR=TENSORRT_DIR    &&\
    make -j$(nproc) && make install && cd .. &&\
    cd mmdeploy &&\
    python3 -m pip install --upgrade setuptools &&\
    python3 -m pip install -e . &&\
    mkdir -p build && cd build &&\
    cmake .. \
    	-DMMDEPLOY_BUILD_SDK=ON \
    	-DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
    	-DMMDEPLOY_BUILD_EXAMPLES=ON \
    	-DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
    	-DMMDEPLOY_TARGET_BACKENDS="trt" \
    	-DTENSORRT_DIR=TENSORRT_DIR \ 
    	-Dpplcv_DIR=/root/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl \
    	-DMMDEPLOY_CODEBASES=all && \
    make -j$(nproc) && make install

# add the patch to solve the torch.distributed issue in docker
RUN cd mmdeploy/tools &&\
    echo "import torch.distributed" > jetson_patch.py && \
    echo "if not torch.distributed.is_available():" >> jetson_patch.py && \
    echo "     torch.distributed.ReduceOp = lambda: None" >> jetson_patch.py && \
    sed -i '2i import jetson_patch' deploy.py

# add patch to solve the build error
RUN sed -i '/def _run_symbolic_method(g, op_name, symbolic_fn, args):/,/except TypeError as e:/\
   {
       s/\
       return symbolic_fn(g, \*args)/\
       graph_context = jit_utils.GraphContext(\
            graph=g,\
            block=g.block(),\
            opset=GLOBALS.export_onnx_opset_version,\
            original_node=None,  # type: ignore[arg-type]\
            params_dict=_params_dict,\
            env={},\
       )\
       return symbolic_fn(graph_context, \*args)/\
    }' /usr/local/lib/python3.8/dist-packages/torch/onnx/symbolic_helper.py

ENV MMDeploy_DIR="/root/workspace/mmdeploy/build/install/lib/cmake/MMDeploy"
ENV LD_LIBRARY_PATH="/root/workspace/mmdeploy/build/lib:${BACKUP_LD_LIBRARY_PATH}"
ENV PATH="/root/workspace/mmdeploy/build/bin:${PATH}"
ENV PYTHONPATH="/root/workspace/mmdeploy:${PYTHONPATH}"
