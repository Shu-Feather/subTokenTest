from torch import nn

class Prober(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int, layer: int = 1, hidden_dim: int = 768):
        super().__init__()
        if layer < 1:
            raise ValueError('layer must be greater than 0')
        if layer == 1:
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            for _ in range(layer - 2):
                self.layers.extend([nn.GELU(), nn.Linear(hidden_dim, hidden_dim)])
            self.layers.extend([nn.GELU(), nn.Linear(hidden_dim, output_dim)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ProberNumber(nn.Module):
    
    def __init__(self, input_dim: int, max_num_each_char: int, char_num: int, layer: int = 1, hidden_dim: int = 768):
        super().__init__()
        self.char_num = char_num
        self.max_num_each_char = max_num_each_char
        output_dim = char_num * (max_num_each_char + 1)
        self.inner_model = Prober(input_dim, output_dim, layer, hidden_dim)

    def forward(self, x):
        x = self.inner_model(x)
        x = x.view(x.shape[0], (self.max_num_each_char + 1), self.char_num) # (batch_size, max_num_each_char + 1, char_num)
        return x
    
class ProberSeq(nn.Module):
    
    def __init__(self, input_dim: int, max_length_each_token: int, char_num: int, layer: int = 1, hidden_dim: int = 768):
        super().__init__()
        self.char_num = char_num
        self.max_length_each_token = max_length_each_token
        output_dim = (char_num + 1) * max_length_each_token
        self.inner_model = Prober(input_dim, output_dim, layer, hidden_dim)

    def forward(self, x):
        x = self.inner_model(x)
        x = x.view(x.shape[0], (self.char_num + 1), self.max_length_each_token) # (batch_size, char_num, max_length_each_token)
        return x