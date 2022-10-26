import random
import argparse

parser = argparse.ArgumentParser()
seed = random.SystemRandom()

parser.add_argument('--n_len', type=int, required=True)
args = parser.parse_args()
sk = seed.getrandbits(args.n_len)
print(sk)