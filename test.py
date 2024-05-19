import torch
import sys

def human_size(n, factor=1000, frac=0.8, prec=1):
    """
    :param int|float n:
    :param int factor: for each of the units K, M, G, T
    :param float frac: when to go over to the next bigger unit
    :param int prec: how much decimals after the dot
    :return: human readable size, using K, M, G, T
    :rtype: str
    """
    postfixes = ["", "K", "M", "G", "T"]
    i = 0
    while i < len(postfixes) - 1 and n > (factor ** (i + 1)) * frac:
        i += 1
    if i == 0:
        return str(n)
    return ("%." + str(prec) + "f") % (float(n) / (factor**i)) + postfixes[i]


def human_bytes_size(n, factor=1024, frac=0.8, prec=1):
    """
    :param int|float n:
    :param int factor: see :func:`human_size`. 1024 by default for bytes
    :param float frac: see :func:`human_size`
    :param int prec: how much decimals after the dot
    :return: human readable byte size, using K, M, G, T, with "B" at the end
    :rtype: str
    """
    return human_size(n, factor=factor, frac=frac, prec=prec) + "B"

print(torch.__version__)
print(torch.cuda.is_available())
dev = torch.device("cuda")
free, total = torch.cuda.mem_get_info()
alloc = torch.cuda.max_memory_allocated()
print(
    f"Total GPU memory {human_bytes_size(total)},"
    f" alloc {human_bytes_size(alloc)},"
    f" free {human_bytes_size(free)} ({free / total * 100:.1f}%)"
)