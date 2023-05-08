from utils import *

SEQ_DIR = '/home/qhoang/Code/masked-minimizer/seqdata/'
chr = 'ATGC'

def generate_partial_digestion_instance(seq_name, k, save_dir=None, min_freq=10, binsize=1e-4):
    lines = open(f'{SEQ_DIR}{seq_name}.seq', 'r').readlines()
    seq = ''.join([line.strip() for line in lines])
    kmer_dict = {}
    bar = trange(len(seq) - k)
    for i in bar:
        kmer = seq[i: i + k]
        if kmer not in kmer_dict:
            kmer_dict[kmer] = [i]
        else:
            kmer_dict[kmer].append(i)
    freq = -1
    while freq < min_freq:
        cutting_kmer = np.random.choice(list(kmer_dict.keys()))
        x = kmer_dict[cutting_kmer]
        freq = len(x)

    dfreq = {}
    for i in range(len(x)):
        for j in range(i, len(x)):
            dij = abs(x[i] - x[j])
            if dij not in dfreq:
                dfreq[dij] = 1
            else:
                dfreq[dij] += 1

    d = list(dfreq.keys())
    d.sort()

    if save_dir is not None:
        # Record points
        with open(f'{save_dir}/{cutting_kmer}_{seq_name}_{len(x)}frags.pts', 'w') as f:
            f.write(' '.join([str(xi) for xi in x]))
        # Record distances
        with open(f'{save_dir}/{cutting_kmer}_{seq_name}_{len(x)}frags.dst', 'w') as f:
            for di in d:
                f.write(f'{di} {dfreq[di]}\n')

    return cutting_kmer, x, dfreq

n_test = 20

for i in range(n_test):
    generate_partial_digestion_instance('chrXC', 15, save_dir='./data', min_freq=10)
