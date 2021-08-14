from smeagol.utils import *
from collections import Counter


def test_equals():
    assert not equals(1.01, 1.012)
    assert equals(1.01, 1.01002)
    assert not equals(np.array([1., 2., 3.]), np.array([1., 2., 2.999]))


records = [SeqRecord(seq=Seq('ATTAAA'), id='S1.1', name='S1'),
           SeqRecord(seq=Seq('GCTATA'), id='S1.2', name='S1')]

def test_shuffle_records():
    def dinuc_freqs(x):
        dinucs = sorted([x[i:i+2] for i in range(len(x) - 1)])
        return Counter(dinucs)
    ref_freqs = [dinuc_freqs(x.seq.__str__()) for x in records]
    shuf = shuffle_records(records, 3, 2)
    assert len(shuf) == 6
    shuf_freqs = [dinuc_freqs(x.seq.__str__()) for x in shuf]
    assert shuf_freqs[0] == shuf_freqs[1] == shuf_freqs[2] == ref_freqs[0]
    assert shuf_freqs[3] == shuf_freqs[4] == shuf_freqs[5] == ref_freqs[1]