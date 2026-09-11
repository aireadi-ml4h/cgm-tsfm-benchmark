"""Reproduces the meal-detection architecture actually implemented in
code/external/SSM-CGM/MealDetection/mealdetection.ipynb's CNN_Dilated_BiLSTM_Attn cell
-- same class name in their notebook, but no attention layer is actually
defined or applied in their forward() (the self.attn call is commented out
and self.attn is never assigned in __init__), so this reproduces the real
executed architecture (dilated CNN -> BiLSTM -> LayerNorm -> Linear), not
the name. Hyperparameters (c_out=16, h=64, dropout=0.4, 2-layer bidirectional
LSTM, kernel/dilation schedule 15/1, 3/2, 3/4) are copied unchanged from
their param dict and cnn Sequential -- see cgmacros_dataset.py's module
docstring for why this project's version runs on 5-minute bins instead of
their apparent 1-minute bins.
"""
import torch.nn as nn


class MealDetector(nn.Module):
    def __init__(self, c_out=16, h=64, dropout=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, c_out, 15, padding=7, dilation=1), nn.ReLU(),
            nn.Conv1d(c_out, c_out, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv1d(c_out, c_out, 3, padding=4, dilation=4), nn.ReLU(),
            nn.BatchNorm1d(c_out),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            c_out, h, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(2 * h)
        self.cls = nn.Linear(2 * h, 1)

    def forward(self, x):
        # x: (B, L, 1)
        x = self.cnn(x.transpose(1, 2))  # (B, C, L)
        x = x.transpose(1, 2)  # (B, L, C)
        x, _ = self.lstm(x)  # (B, L, 2h)
        x = self.norm(x)
        return self.cls(x).squeeze(-1)  # (B, L) logits
