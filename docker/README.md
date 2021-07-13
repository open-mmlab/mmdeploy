## Use the container
Place the Dockerfile and the [TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.3/tars/TensorRT-7.2.3.4.Ubuntu-16.04.x86_64-gnu.cuda-11.1.cudnn8.1.tar.gz) package togather under a folder named docker.
Run the following command to build the docker image:
```
sudo docker build docker/ -t mmdeploy
```
Then run the command bellow to play with the docker image:
```
sudo docker run --gpus all --shm-size=8g -it -p 8084:8084 mmdeploy
```

## Use the optimized container
The optimized docker file is provided and it can be used in the future.
