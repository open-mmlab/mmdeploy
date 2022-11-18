# from mmdeploy.backend.ipu.wrapper import IPUWrapper
import model_runtime
import numpy as np
import popef
import time
from datetime import timedelta

popef_file = '/localdata/cn-customer-engineering/qiangg/projects/byte-mlperf-1/session_cache/2272664960880696850.popef'
# 64562147318042298.popef  2272664960880696850

config = model_runtime.ModelRunnerConfig()
config.device_wait_config = model_runtime.DeviceWaitConfig(
    model_runtime.DeviceWaitStrategy.WAIT_WITH_TIMEOUT,
    timeout=timedelta(microseconds=0),
    sleepTime=timedelta(seconds=0))

print("Creating ModelRunner with", config)
popef_path = model_runtime.PopefPaths(popef_file)
print('popef path ', popef_path)
runner = model_runtime.ModelRunner(popef_file,
                                   config=config)

print("created the runner")
input_descriptions = runner.getExecuteInputs()
print('input description ', input_descriptions)

input_tensors = [
    np.random.randn(*input_desc.shape).astype(
        popef.popefTypeToNumpyDType(input_desc.data_type))
    for input_desc in input_descriptions
]
input_view = model_runtime.InputMemoryView()

bps = 1

for input_desc, input_tensor in zip(input_descriptions, input_tensors):
    print("\tname:", input_desc.name, "shape:", input_tensor.shape,
          "dtype:", input_tensor.dtype)
    input_tensor = np.repeat(input_tensor, repeats=bps, axis=0)
    input_view[input_desc.name] = input_tensor

# output_future_memory = model_runtime.OutputFutureMemory()

for i in range(100):
    start = time.time()

    result = runner.executeAsync(input_view)
result.wait()
print('time elapsed ', time.time()-start)
output_descriptions = runner.getExecuteOutputs()
print('output description ', output_descriptions)

outputs = {}
print("Processing output tensors:")
for output_desc in output_descriptions:
    out_shape = output_desc.shape
    out_shape[0] = out_shape[0] * bps
    output_tensor = np.frombuffer(result[output_desc.name],
                                  dtype=popef.popefTypeToNumpyDType(
        output_desc.data_type)).reshape(
        out_shape)
    outputs[output_desc.name] = output_tensor


print(outputs)

# wp = IPUWrapper(popoef_file=popef_file)
