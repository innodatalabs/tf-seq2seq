import random

VOCAB = 'abcdefghijklmopqrstuvqxyz'

def random_string(length):
    out = []
    
    for _ in range(length):
        out.append(VOCAB[random.randint(0, len(VOCAB)-1)])

    return ''.join(out)


if __name__ == '__main__':

    # Some easy-to-learn data patterns
    for _ in range(100000):
        print(random_string(10))
