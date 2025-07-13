# @title bptt train, gen
import torch
from torch.nn import functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
scaler = torch.GradScaler()

def Perplexity(logits, target): # [b,t,vocab_size], [b,t]
    log_probs = F.log_softmax(logits, dim=-1)
    # nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1) # [b,t]
    nll = -log_probs.gather(dim=-1, index=target.to(torch.int64).unsqueeze(-1)).squeeze(-1) # [b,t]
    perplexity = nll.mean().exp()
    return perplexity

import time
def strain(model, dataloader, optim, scheduler=None, bptt=25): # train function with automatic mixed precision
    model.train()
    h0 = None
    start = begin = time.time()
    for i, x in enumerate(dataloader):
        x = x.to(device)
        xs, ys = torch.split(x[:,:-1], bptt, dim=1), torch.split(x[:,1:], bptt, dim=1)
        for (x, y) in zip(xs, ys): # https://discuss.pytorch.org/t/how-to-train-a-many-to-many-lstm-with-bptt/13005/10
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                pred, h0 = model(x, h0)
                loss = F.cross_entropy(pred.flatten(0,1), y.flatten().to(int)) # [b*t,d], [b*t]
                # loss = F.cross_entropy(pred.flatten(0,1), y.flatten()) # [b*t,d], [b*t]
            optim.zero_grad()
            scaler.scale(loss).backward()
            # scaler.unscale_(optim)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optim)
            scaler.update()
            h0 = h0.detach()

        # if scheduler is not None: scheduler.step()
        if i % 100 == 0:
            perplexity = Perplexity(pred.detach(), y).item()
            print("strain loss, ppl",loss.item(), perplexity)
            print(generate(model, "this is what"))
            model.train()
            print(i, 'time:',time.time() - start, (time.time()-begin)/(i+1))
            start = begin = time.time()
        try: wandb.log({"train loss": loss.item()/len(y)})
        except NameError: pass

def generate(model, context, max_steps=64, temperature=1):
    x = encode(context)#.to(device)
    model.eval()
    hidden=None # rnn 1/3
    for n in range(max_steps):
        with torch.no_grad():
            # output = model(x) # gpt
            output, hidden = model(x, hidden) # rnn 2/3
        hidden = hidden[:,-1].unsqueeze(1) # rnn 3/3
        output = output[:,-1] # get logit for last character
        output = output/temperature
        output = F.softmax(output, dim=-1) # vocab_size to char
        ix = torch.multinomial(output, num_samples=1) # rand sample by output distribution
        x = torch.cat((x, ix), dim=1)
    completion = decode(x.squeeze(0))
    # completion = decode(x)
    return completion

# import time
# start = begin = time.time()
for i in range(1):
    # train_loss = strain(model, train_loader, optim, scheduler=None)
    strain(model, train_loader, optim, scheduler=None)
    print(generate(model, "this is what"))
    # print(i, 'time:',time.time() - start, (time.time()-begin)/(i+1))
    # start = time.time()
