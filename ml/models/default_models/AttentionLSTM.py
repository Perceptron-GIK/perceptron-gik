from torch import nn
import torch
class AttentionLSTM(nn.Module):
    """LSTM with self-attention mechanism (inspired by GloveTyping paper)."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        num_heads: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.projection = nn.Linear(lstm_out_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.layer_norm(lstm_out + attn_out)
        
        if self.bidirectional:
            last_out = torch.cat([out[:, -1, :self.hidden_dim], 
                                  out[:, 0, self.hidden_dim:]], dim=-1)
        else:
            last_out = out[:, -1, :]
        
        return self.projection(last_out)