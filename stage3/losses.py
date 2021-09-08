import torch
import torch.nn as nn
import torch.nn.functional as F

class PANNsLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.cel = nn.CrossEntropyLoss()

    def forward(self, input, target):
        """
        input_ = input
        input_ = torch.where(
            torch.isnan(input_),
            torch.zeros_like(input_),
            input_
        )
        input_ = torch.where(
            torch.isinf(input_),
            torch.zeros_like(input_),
            input_
        )

        target = target.float()
        """

        #return self.bce(input_, target)
        return self.bce(input, target)