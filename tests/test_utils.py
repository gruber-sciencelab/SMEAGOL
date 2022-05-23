from smeagol.utils import *
from collections import Counter


def test_equals():
    assert not _equals(1.01, 1.012)
    assert _equals(1.01, 1.01002)
    assert not _equals(np.array([1., 2., 3.]), np.array([1., 2., 2.999]))

    
genome = [SeqRecord(seq=Seq('AGAGCGCATTTCCTACGCATGCTCGATCAAATGCTACGGATTCTAAAA'), id='seg1', name='seg1'),
           SeqRecord(seq=Seq('CCCGGACGTTTGCTCGAGGG'), id='seg2', name='seg2')]


def test_shuffle_records():
    def dinuc_freqs(x):
        dinucs = sorted([x[i:i+2] for i in range(len(x) - 1)])
        return Counter(dinucs)
    ref_freqs = [dinuc_freqs(x.seq.__str__()) for x in genome]
    shuf = shuffle_records(genome, 3, 2)
    assert len(shuf) == 6
    shuf_freqs = [dinuc_freqs(x.seq.__str__()) for x in shuf]
    assert shuf_freqs[0] == shuf_freqs[1] == shuf_freqs[2] == ref_freqs[0]
    assert shuf_freqs[3] == shuf_freqs[4] == shuf_freqs[5] == ref_freqs[1]