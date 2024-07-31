import torch
import torch.nn as nn
import torch.optim as optim


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
    
class LatentDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(LatentDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def generate_mask(self, latents, frame):
        position_p = torch.bernoulli(torch.Tensor([1 - self.p]*(latents//2)))
        return position_p.repeat(1, frame, 2)


    def forward(self, x: torch.Tensor):
        if self.training:
            _, frame, latent = x.size() 
            landmark_mask = self.generate_mask(latent, frame)
            scale = 1/(1-self.p)
            return x*landmark_mask.to(x.device)*scale
        else:
            return x
        

class StyleGRU(nn.Module):
    def __init__(self,
                 feature_size=9216,
                 lm_dropout_rate=0.2,   
                 rnn_unit=4096,         
                 num_layers=2,
                 rnn_dropout_rate=0.1,  
                 fc_dropout_rate=0.5,   
                 res_hidden=4096,
                 ):
        
        super(StyleGRU, self).__init__()
        self.hidden_size = rnn_unit
        self.hidden_state = nn.Parameter(torch.randn(2 * num_layers, 1, rnn_unit)) 
        self.dropout_latent = LatentDropout(lm_dropout_rate)
        
        self.gru = nn.GRU(input_size=feature_size, hidden_size=rnn_unit,
                          num_layers=num_layers, dropout=rnn_dropout_rate,
                          batch_first=True, bidirectional=True)

        self.dense = nn.Sequential(
            nn.Dropout(fc_dropout_rate),
            Residual(FeedForward(rnn_unit * 2 * num_layers, res_hidden, fc_dropout_rate)),
            nn.Dropout(fc_dropout_rate),
            nn.Linear(rnn_unit * 2 * num_layers, 1) 
        )
        
        self.bn = nn.BatchNorm1d(31) 

    def forward(self, x):
        self.gru.flatten_parameters()
        x = self.dropout_latent(x)
        x = self.bn(x)
        _, hidden = self.gru(x, self.hidden_state.repeat(1, x.shape[0], 1))
        x = torch.cat(list(hidden), dim=1)
        hidden_ = x.clone()
        x = self.dense(x)
        
        return x, hidden_