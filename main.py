import torch
import time

from names_dataset import NamesDataset
from char_rnn import CharRNN
from train import train
from evaluate import evaluate, plot_losses
import util

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
torch.set_default_device(device)

print(f"Using device {torch.get_default_device()}")

alldata = NamesDataset("data/names")
train_set, test_set = torch.utils.data.random_split(alldata, [.85, .15],
                                                    generator=torch.Generator(device=device).manual_seed(2024))

print(f"Loaded {len(alldata)} items of data with a split of {len(train_set)} train and {len(test_set)} test")

n_hidden = 128
rnn = CharRNN(util.n_letters(), n_hidden, len(alldata.labels_uniq))

start = time.time()
all_losses = train(rnn, train_set, n_epoch=27, learning_rate=0.15, report_every=5)
end = time.time()
print(f"Completed training in {(end - start) * 1000:.1f}ms")

plot_losses(all_losses)
evaluate(rnn, test_set, classes=alldata.labels_uniq)
