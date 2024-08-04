# based on: https://github.com/karpathy/llama2.c/blob/master/export.py
import struct

import torch
from os import path as osp


def main():
    try:
        # available scales are 0.5 and 1.0
        net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
    except RuntimeError as _:
        path = osp.expanduser("~/.cache/torch/hub/checkpoints/unet_carvana_scale0.5_epoch2.pth")
        state_dict = torch.load(path, weights_only=False, map_location='cpu')
    else:
        state_dict = net.to("cpu").state_dict()
    
    with open("weights.bin", "wb") as out:
        for k, v in state_dict.items():
            if "num_batches_tracked" in k:
                continue
            assert(v.dtype == torch.float32)
            w = v.detach().view(-1).numpy()
            b = struct.pack(f"{len(w)}f", *w)
            out.write(b)

if __name__ == '__main__':
    main()
