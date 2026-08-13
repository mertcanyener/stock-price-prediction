import torch
import torch.nn as nn

INPUT_SIZE = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


class StockLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        out = self.dropout(last_step)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    model = StockLSTM()
    print(model)

    dummy_input = torch.randn(8, 30, 5)
    output = model(dummy_input)

    print(f"\nDummy input shape: {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    assert output.shape == (8, 1), f"Beklenen (8, 1), gelen {tuple(output.shape)}"
    print("Output shape dogrulandi: (8, 1)")
