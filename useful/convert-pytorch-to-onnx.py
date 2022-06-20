import torch

from training.DeepEmotion import DeepEmotion
from training.DAN import DAN
from training.DACL import *

# Deep Emotion torch.size [128, 1, 48, 48]
# DAN torch.size [213, 3, 224, 224]
# DACL torch.size [128, 3, 224, 224]
# DACLAll torch.size [64, 3, 224, 224]


def main():
    name = "DACL_FERG_AffectNet_RAF-DB"
    pytorch_model = DACL(BasicBlock, [2, 2, 2, 2])
    pytorch_model.load_state_dict(torch.load(
        f'training/Checkpoints/{name}.pt'))
    pytorch_model.eval()
    dummy_input = torch.zeros([128, 3, 224, 224])
    torch.onnx.export(pytorch_model, dummy_input,
                      f'{name}.onnx', verbose=True)


if __name__ == '__main__':
    main()
