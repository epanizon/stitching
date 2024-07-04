import torch

def overlap(p1s, p2s):
    ov = torch.minimum(p1s, p2s).sum()
    return ov.to('cpu').detach().numpy()


def best_overlap(list1, list2, N):
    return len(set(list1).intersection(list2)) / N


def best_guesses(logits, tokenizer, N=5):
    momo = torch.softmax(logits, 0)
    momo = torch.argsort(momo, descending=True)[:N]
    momo = [tokenizer.decode(m) for m in momo]
    return momo


def entropy_from_logits(logits):
    logits = logits - torch.max(logits)
    p = torch.exp(logits)
    p /= p.sum()
    return -(p*torch.log(p)).sum()

@torch.no_grad()
def correct_guess(logits, correct, N=5):
    momo = torch.softmax(logits, 0)
    momo = torch.argsort(momo, descending=True)[:N]
    cc = torch.zeros(N)
    cc[momo == correct] = 1.
    return torch.cumsum(cc, dim=0)

def print_best_guesses(logits, correct, tokenizer, N=5):
    momo = torch.softmax(logits, 0)
    momo = torch.argsort(momo, descending=True)[:N]
    print("Guesses: ", tokenizer.decode(momo), " . Correct: ", tokenizer.decode(correct))
    return None
