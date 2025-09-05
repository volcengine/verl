import io
import torch

def chunk_list(lst, n_chunks):
    """Split a list into chunks of equal length"""
    size = len(lst) // n_chunks
    for i, start in enumerate(range(0, len(lst), size)):
        if i == n_chunks - 1:
            yield lst[start : ]
            return
        else:
            yield lst[start : start + size]


def serialize(data):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    return buffer.getbuffer()


def deserialize(message):
    buffer = io.BytesIO(message)
    return torch.load(buffer)


if __name__ == "__main__":
    lst = list(range(12))
    sub_lsts = list(chunk_list(lst, 3))

    assert len(sub_lsts) == 3
    assert sub_lsts[0] == [0, 1, 2, 3]
    assert sub_lsts[1] == [4, 5, 6, 7]
    assert sub_lsts[2] == [8, 9, 10, 11]

    lst = list(range(11))
    sub_lsts = list(chunk_list(lst, 3))
    assert len(sub_lsts) == 3
    assert sub_lsts[0] == [0, 1, 2]
    assert sub_lsts[1] == [3, 4, 5]
    assert sub_lsts[2] == [6, 7, 8, 9, 10]

    lst = list(range(11))
    sub_lsts = list(chunk_list(lst, 1))
    assert len(sub_lsts) == 1
    assert sub_lsts[0] == list(range(11))