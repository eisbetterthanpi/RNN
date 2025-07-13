# @title GRU pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"

class RNN(nn.Module):
    def __init__(self, in_dim, d_model, out_dim=None, num_layers=1):
        super().__init__()
        if out_dim is None: out_dim = in_dim
        self.num_layers = num_layers
        self.d_model = d_model
        self.emb = nn.Embedding(in_dim, d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers, batch_first=True)
        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, x, h0=None): # rnn/gru
        x = self.emb(x)
        if h0==None: h0 = torch.zeros((self.num_layers, x.size(0), self.d_model), device=device)
        # print(x.shape, h0.shape)
        x, h0 = self.rnn(x, h0)
        # out = out[:, -1, :] # out: (n, 128)
        x = self.fc(x) # out: (n, 10)
        return x, h0

hidden_size = 128 #64 128
num_layers = 3#2
try: vocab_size=train_loader.dataset.vocab_size#50
except NameError: vocab_size=50

model = RNN(vocab_size, hidden_size, vocab_size, num_layers).to(device)
# print(model)
print(sum(p.numel() for p in model.parameters() if p.requires_grad)) # 19683
optim = torch.optim.AdamW(model.parameters(), 1e-3)


# 128,2
# Test Loss: 6.360389362062727
# this is what ween new york is well it a sign more directly into simply shares
# 0 time: 5.910429954528809 5.910431623458862
# 64,2
# Test Loss: 6.167358561924526
# this is what bull morp on its aftant opereals of a b. the hamally plans beces
# 0 time: 5.655158996582031 5.655160903930664
# 64,1
# Test Loss: 5.823708357129778
# this is what achan agrive
#  the guinesst on of promjects cl funds that jound
# 0 time: 4.788918495178223

# Test Loss: 8.247572830745153
# this is what that has months with unfinsings lide to N by well
#  next offited
# 29 time: 3.902247905731201 4.377542002995809

# dim128 3lyr bptt25 2.8,.2
# this is whatever from the every coddane
# teets, in.]
# [Footnote 21: Any ot
# 9700 time: 4.124014616012573 0.00042511236912171237
# strain loss, ppl 1.3780442476272583 3.953125
# this is what seeming upon course in your drew
# plant now, as a less ever aff

b,t=2,5
x = torch.randint(0, vocab_size, (b,t), device=device)
h = torch.rand(num_layers, b, hidden_size, device=device)
out, h = model(x, h)
print(out.shape, h.shape)
