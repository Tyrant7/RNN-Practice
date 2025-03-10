import torch
import string
import unicodedata

allowed_characters = string.ascii_letters + " ..;"


def n_letters():
    return len(allowed_characters)


# Turn a Unicode string to plain ASCII
def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
        and c in allowed_characters
    )


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return allowed_characters.find(letter)


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters())
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor


def label_from_output(output, output_labels):
    top_n, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i
