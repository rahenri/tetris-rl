import hashlib


def hash_value(value):
    return hashlib.sha256(str(value).encode()).hexdigest()
