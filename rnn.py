import torch
import torch.nn as nn

class AUGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias = True):
        super(AUGRUCell, self).__init__()

        in_dim = input_dim + hidden_dim
        self.reset_gate = nn.Sequential( nn.Linear( in_dim, hidden_dim, bias = bias), nn.Sigmoid())
        self.update_gate = nn.Sequential( nn.Linear( in_dim, hidden_dim, bias = bias), nn.Sigmoid())
        self.h_hat_gate = nn.Sequential( nn.Linear( in_dim, hidden_dim, bias = bias), nn.Tanh())


    def forward(self, X, h_prev, attention_score):
        temp_input = torch.cat( [ h_prev, X ] , dim = -1)
        r = self.reset_gate( temp_input)
        u = self.update_gate( temp_input)

        h_hat = self.h_hat_gate( torch.cat( [ h_prev * r, X], dim = -1) )

        u = attention_score.unsqueeze(1) * u
        h_cur = (1. - u) * h_prev + u * h_hat

        return h_cur


class DynamicGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_cell = AUGRUCell( input_dim, hidden_dim, bias = True)

    def forward(self, X, attenion_scores , h0 = None ):
        B, T, D = X.shape
        H = self.hidden_dim
        
        output = torch.zeros( B, T, H ).type( X.type() )
        h_prev = torch.zeros( B, H ).type( X.type() ) if h0 == None else h0
        for t in range( T): 
            h_prev = output[ : , t, :] = self.rnn_cell( X[ : , t, :], h_prev, attenion_scores[ :, t] )
        return output
