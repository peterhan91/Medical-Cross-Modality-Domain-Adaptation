import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # intersection = torch.sum(input_soft * target, dims)
        # cardinality = torch.sum(input_soft + target, dims)
        tp = torch.sum(input_soft * target, dims)
        fn = torch.sum((1.-input_soft) * target, dims)
        fp = torch.sum(input_soft * (1 - target), dims)

        # dice_score = -1.*((2.*intersection+1.) / (cardinality+1.))
        dice = -((2 * tp + 1.) / (2 * tp + fp + fn + 1.))
        return torch.mean(dice)