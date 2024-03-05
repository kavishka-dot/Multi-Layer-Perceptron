import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

# read the data set
words = open('names.txt','r').read().splitlines()

# Build a vocabulary of characters and a mapping from/to integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

#data set
blockSize = 3 #context window size

def build_dataset(words):

  X,Y = [], [] #inputs and targets
  for w in words:
    context = [0] * blockSize
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] #crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X, Y

import random
random.seed(42)
random.shuffle(words) #randomly shuffle the words
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])       #80%
Xdev, Ydev = build_dataset(words[n1:n2])   #10%
Xte, Yte = build_dataset(words[n2:])       #10%

n_embed = 10   # The dimensionality of the character embedding vectors
n_hidden = 200 # The number of neurons in the hidden layer

g = torch.Generator().manual_seed(2147483647) #for reproducibility

C = torch.randn((vocab_size,n_embed),              generator = g)
W1 = torch.randn((n_embed * blockSize , n_hidden), generator = g)
b1 = torch.randn(n_hidden,                         generator = g)
W2 = torch.randn((n_hidden,vocab_size),            generator = g)
b2 = torch.randn(vocab_size,                       generator = g)

parameters = [C, W1, b1 ,W2, b2]
print(sum(p.nelement() for p in parameters)) #number of parameters in total

for p in parameters:
  p.requires_grad = True
  
# optimization
max_steps = 200000
batch_size = 32
lossi =[]

for i in range(max_steps):
  #mini batch construct
  ix = torch.randint(0,Xtr.shape[0],(batch_size,))

  #forward pass
  emb = C[Xtr[ix]]
  h = torch.tanh(emb.view(-1, n_embed * blockSize) @ W1 + b1)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Ytr[ix])

  #backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  #update
  lr = 0.1 if i<10000 else 0.01
  for p in parameters:
    p.data += -lr*p.grad

  lossi.append(loss.log10().item())
  
@torch.no_grad()  # this decorator disables gradient tracking
def split_loss(split):
  x,y = {
      'train': (Xtr,Ytr),
      'val' : (Xdev,Ydev),
      'test': (Xte,Yte)
  }[split]
  emb = C[x]
  h = torch.tanh(emb.view(-1, n_embed * blockSize) @ W1 + b1)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, y)
  print(split, f"{loss.item() : .4f}")

split_loss('train')
split_loss('val')

# sample from the distribution
g = torch.Generator().manual_seed(2147483647+10)

for _ in range(10):

  out = []
  context = [0]* blockSize 

  while True:

    # forward pass
    emb = C[torch.tensor([context])]
    h = torch.tanh(emb.view(1, -1) @ W1 + b1)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1)

    # sample from the distribution
    ix = torch.multinomial(probs, num_samples=1, generator = g).item()

    #shift the context window
    context = context[1:] + [ix]
    out.append(ix)
    if ix == 0:
      break
      
  print(''.join(itos[i] for i in  out))
