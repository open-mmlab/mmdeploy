# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function

import logging

import grpc
import inference_pb2
import inference_pb2_grpc
import os
import cv2
import numpy as np

def build_dummy_tensor():
    img = cv2.imread('/home/PJLAB/konghuanjun/Downloads/snpe-1.55.0.2958/models/alexnet/data/chairs.jpg')
    m = cv2.resize(img, (224, 224))
    data = (m.astype(np.float32) - 127.5) / 127.5
    print(data.shape)
    tensor = inference_pb2.Tensor(data=data.tobytes(), shape=list(data.shape), name='data_0', dtype='float32')
    return tensor

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    filename = 'end2end.dlc'
    filesize = os.stat(filename).st_size
    
    weights = bytes()
    # with open(filename, 'rb') as f:
    #     weights = f.read(filesize)
    # if len(weights) >= (2 << 29):
    #     print('model size too big')
        
    # https://github.com/grpc/grpc/blob/v1.46.x/include/grpc/impl/codegen/grpc_types.h
    # https://grpc.io/docs/guides/performance/
    with grpc.insecure_channel('10.1.80.67:50051', 
                               options=(
                                   ('grpc.GRPC_ARG_KEEPALIVE_TIME_MS', 2000),
                                   ('grpc.max_send_message_length', 2<<29),
                                   ('grpc.keepalive_permit_without_calls', 1))) as channel:
        print("channel type {}".format(type(channel)))
        # with grpc.insecure_channel('[0:0:fe80::3455:bf2a]:50051') as channel:
        stub = inference_pb2_grpc.InferenceStub(channel)
        response = stub.Echo(inference_pb2.Empty())
        print("Response echo {}".format(response))
        
        model = inference_pb2.Model(name= filename, weights=weights, device=1)
        print("Sending model to init, please wait...")
        response = stub.Init(model)
        print("Response init {}".format(response))
        
        response = stub.OutputNames(inference_pb2.Empty())
        print("Response outputnames {}".format(response))
        
        tensor = build_dummy_tensor()
        tensorList = inference_pb2.TensorList(datas = [tensor])
        
        for x in range(1):
            response = stub.Inference(tensorList)
            if response.status == 0:
                prob = np.frombuffer(response.datas[0].data, dtype=np.float32)
                print("prob argmax: {} max: {}".format(prob.argmax(), prob.max()))
            else:
                print(response.info)

if __name__ == '__main__':
    logging.basicConfig()
    run()
