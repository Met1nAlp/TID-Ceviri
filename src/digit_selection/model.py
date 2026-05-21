"""Small MLP for digit-based selection."""

import torch
import torch.nn as nn

from src.digit_selection.config import INPUT_FEATURES, NUM_CLASSES


class DigitSelectionMLP(nn.Module):
    """Compact classifier for 63-dimensional hand landmark vectors."""

    def __init__(
        self,
        input_size: int = INPUT_FEATURES,
        hidden_sizes=(128, 64),
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.25,
    ):
        super().__init__()

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, num_classes))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    model = DigitSelectionMLP()
    x = torch.randn(4, INPUT_FEATURES)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
