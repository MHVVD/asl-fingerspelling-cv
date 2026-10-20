"""Training for the sequence model (PyTorch, CPU). Runtime inference is NumPy-only (seqmodel.SeqModel).

The network:  Linear+ReLU -> bidirectional GRU -> Linear, trained with CTC loss on (features, word) pairs.
"""
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class SeqNet(nn.Module):
    def __init__(self, n_in, n_out, hidden=96, layers=2, dropout=0.2):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, bidirectional=True, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(2 * hidden, n_out)

    def forward(self, x, lengths=None):
        h = torch.relu(self.inp(x))
        if lengths is not None:      # pack so the backward direction starts at each sequence's real end
            h = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
            h, _ = self.gru(h)
            h, _ = pad_packed_sequence(h, batch_first=True)
        else:
            h, _ = self.gru(h)
        return self.out(h)


def ctc_batch_loss(net, feats, targets, ctc):
    """feats: list of (T_i, F) arrays; targets: list of id lists. Returns mean CTC loss."""
    lengths = torch.tensor([len(f) for f in feats])
    x = pad_sequence([torch.from_numpy(np.asarray(f, np.float32)) for f in feats], batch_first=True)
    logp = net(x, lengths).log_softmax(-1).transpose(0, 1)          # (T, B, V) for CTCLoss
    y = torch.tensor([i for t in targets for i in t], dtype=torch.long)
    y_len = torch.tensor([len(t) for t in targets])
    return ctc(logp, y, lengths, y_len)


def train(net, train_items, vocab, epochs=30, batch=32, lr=2e-3, augment=None, val_items=None, val_fn=None,
          log=print, seed=0):
    """train_items: list of (features (T,F), text). augment(features, rng) -> features (optional).
    val_fn(net) -> dict of metrics, called after every epoch; the best state by val 'score' is restored."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ctc = nn.CTCLoss(blank=vocab.blank, zero_infinity=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(train_items) + batch - 1) // batch))
    best, best_state, history = -1.0, None, []
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(train_items))
        losses = []
        for i in range(0, len(order), batch):
            items = [train_items[j] for j in order[i:i + batch]]
            feats = [augment(f, rng) if augment else f for f, _ in items]
            loss = ctc_batch_loss(net, feats, [vocab.encode(t) for _, t in items], ctc)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(float(loss))
        row = {"epoch": ep + 1, "loss": float(np.mean(losses))}
        if val_fn:
            net.eval()
            with torch.no_grad():
                row.update(val_fn(net))
            if row.get("score", -1) > best:
                best, best_state = row["score"], {k: v.clone() for k, v in net.state_dict().items()}
        history.append(row)
        log("  " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
    if best_state is not None:
        net.load_state_dict(best_state)
    return history
