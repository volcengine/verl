import os


def delete_files(*file_names):
    for file_name in file_names:
        try:
            os.remove(file_name)
        except OSError:
            pass


def to_bytes(value):
    """Convert an integer or string to bytes."""
    assert isinstance(value, int) or isinstance(value, str), "replay buffer key must be an int or a string."

    if isinstance(value, int):
        byte_length = (value.bit_length() + 7) // 8 or 1  # Calculate byte length
        return value.to_bytes(byte_length, byteorder="big", signed=True)
    elif isinstance(value, str):
        return value.encode("utf-8")

    return None
