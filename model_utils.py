import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Exponential Moving Average (EMA) for model parameters.

        Args:
            model (nn.Module): The model whose parameters will be averaged.
            decay (float): The decay rate for the EMA. Higher values result in slower updates.
        """
        self.model = model
        self.decay = decay
        self.ema_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def update(self):
        """
        Update the EMA parameters using the current model parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_params[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    @torch.no_grad()
    def apply_ema(self):
        """
        Apply the EMA parameters to the model (for validation/inference).
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.ema_params[name])

    @torch.no_grad()
    def restore(self):
        """
        Restore the model's original parameters (after validation/inference).
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.ema_params[name])
