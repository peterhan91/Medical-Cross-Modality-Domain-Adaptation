import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(input, target):
    smooth = 1.
    loss = 0.
    for c in range(n_classes):
           iflat = input[:, c ].view(-1)
           tflat = target[:, c].view(-1)
           intersection = (iflat * tflat).sum()
           loss += -1. * ((2. * intersection + smooth) /
                             (iflat.sum() + tflat.sum() + smooth))
    return loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes=5) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # compute the actual dice score
        dims = (0, 2, 3)
        intersection = torch.sum(input_soft * target, dims)
        cardinality = torch.sum(input_soft + target, dims)

        dice_score = -2. * intersection / (cardinality + self.eps)
        return torch.mean(dice_score)