from typing import Dict, Iterable, Optional

import ncnn
import numpy as np
import torch

from mmdeploy.apis.ncnn import ncnn_ext


class NCNNWrapper(torch.nn.Module):
    """NCNN Wrapper.

    Arguments:
        param_file (str): param file path
        bin_file (str): bin file path
    """

    def __init__(self,
                 param_file: str,
                 bin_file: str,
                 output_names: Optional[Iterable[str]] = None,
                 **kwargs):
        super().__init__()

        net = ncnn.Net()
        ncnn_ext.register_mm_custom_layers(net)
        net.load_param(param_file)
        net.load_model(bin_file)

        self._net = net
        self._output_names = output_names

    def set_output_names(self, output_names):
        self._output_names = output_names

    def get_output_names(self):
        if self._output_names is not None:
            return self._output_names
        else:
            assert hasattr(self._net, 'output_names')
            return self._net.output_names()

    def forward(self, inputs: Dict[str, torch.Tensor]):
        batch_size = next(iter(inputs.values())).size(0)
        for k, v in inputs.items():
            assert v.size(
                0) == batch_size, 'All tensor should have same batch size'
            assert v.device.type == 'cpu', 'NCNN only support cpu device'

        # set output names
        output_names = self.get_output_names()

        # create output dict
        outputs = dict([name, [None] * batch_size] for name in output_names)

        # inference
        for batch_id in range(batch_size):
            # create extractor
            ex = self._net.create_extractor()

            # set input
            for k, v in inputs.items():
                in_data = ncnn.Mat(v[batch_id].detach().cpu().numpy())
                ex.input(k, in_data)

            # get output
            for name in output_names:
                out_ret, out = ex.extract(name)
                assert out_ret == 0, f'output {out} extract failed.'
                outputs[name][batch_id] = torch.from_numpy(np.array(out))

        # stack outputs together
        for k, v in outputs.items():
            outputs[k] = torch.stack(v)

        return outputs
