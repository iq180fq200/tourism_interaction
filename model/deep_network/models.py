import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self,in_dim,h_dim = 500,hidden_layer_numbers = 2):
        super(MLP,self).__init__()
        self.in_layer = nn.Linear(in_dim,h_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(hidden_layer_numbers):
            self.hidden_layers = self.hidden_layers.append(nn.Linear(h_dim,h_dim))
        self.out_layer = nn.Linear(h_dim,1)
        self.activation_f = F.leaky_relu
    def forward(self,input):
        h = self.activation_f(self.in_layer(input))
        for layer in self.hidden_layers:
            h = self.activation_f(layer(h))
        embedding = h
        return self.out_layer(h),embedding

    def get_loss(self,weights,labels):
        predict_loss = F.mse_loss(weights,labels)
        return predict_loss


