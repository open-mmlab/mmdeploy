# from mmdeploy.backend.ipu.wrapper import IPUWrapper
import model_runtime
import numpy as np
import popef

popef_file = '/localdata/cn-customer-engineering/qiangg/cache_poptorch/5299458688024344947.popef'


config = model_runtime.ModelRunnerConfig()
# config.device_wait_config = model_runtime.DeviceWaitConfig(
#     model_runtime.DeviceWaitStrategy.WAIT_WITH_TIMEOUT,
#     timeout=timedelta(seconds=600),
#     sleepTime=timedelta(seconds=1))

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

for input_desc, input_tensor in zip(input_descriptions, input_tensors):
    print("\tname:", input_desc.name, "shape:", input_tensor.shape,
          "dtype:", input_tensor.dtype)
    input_view[input_desc.name] = input_tensor

result = runner.execute(input_view)
output_descriptions = runner.getExecuteOutputs()
print('output description ', output_descriptions)

outputs = {}
print("Processing output tensors:")
for output_desc in output_descriptions:
    output_tensor = np.frombuffer(result[output_desc.name],
                                  dtype=popef.popefTypeToNumpyDType(
        output_desc.data_type)).reshape(
        output_desc.shape)
    outputs[output_desc.name] = output_tensor
    

print(outputs)

# wp = IPUWrapper(popoef_file=popef_file)


