import torch


def pad_with_value(x, pad_dim, pad_size, pad_value=None):
    num_dims = len(x.shape)
    pad_slice = (slice(None, None, None), ) * num_dims
    pad_slice = pad_slice[:pad_dim] + (slice(0, 1,
                                             1), ) + pad_slice[pad_dim + 1:]
    repeat_size = [1] * num_dims
    repeat_size[pad_dim] = pad_size

    x_pad = x.__getitem__(pad_slice)
    if pad_value is not None:
        x_pad = x_pad * 0 + pad_value

    x_pad = x_pad.repeat(*repeat_size)
    x = torch.cat([x, x_pad], dim=pad_dim)
    return x
