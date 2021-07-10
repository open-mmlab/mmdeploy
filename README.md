# MMDeployment

## Installation

- Build backend ops

  - update submodule

    ```bash
    git submodule update --init
    ```

  - Build with onnxruntime support

    ```bash
    mkdir build
    cd build
    cmake -DBUILD_ONNXRUNTIME_OPS=ON -DONNXRUNTIME_DIR=${PATH_TO_ONNXRUNTIME} ..
    make -j10
    ```

  - Build with tensorrt support

    ```bash
    mkdir build
    cd build
    cmake -DBUILD_TENSORRT_OPS=ON -DTENSORRT_DIR=${PATH_TO_TENSORRT} ..
    make -j10
    ```

  - Or you can add multiple flags to build multiple backend ops.

- Setup project

    ```bash
    python setup.py develop
    ```

## Usage

```bash
python ./tools/deploy.py \
    ${DEPLOY_CFG_PATH} \
    ${MODEL_CFG_PATH} \
    ${MODEL_CHECKPOINT_PATH} \
    ${INPUT_IMG} \
    --work-dir ${WORK_DIR} \
    --device ${DEVICE} \
    --log-level INFO
```
