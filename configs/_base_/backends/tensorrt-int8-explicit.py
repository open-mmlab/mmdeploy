_base_ = ['./tensorrt.py']

backend_config = dict(
    common_config=dict(
        fp16_mode=False, int8_mode=True, explicit_quant_mode=True))

function_record_to_pop = ['torch.cat']
