from smeagol.enrich import *
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from smeagol.models import PWMModel
from smeagol.utils import equals

script_dir = os.path.dirname(__file__)
rel_path = "data"
data_path = os.path.join(script_dir, rel_path)

genome = [SeqRecord(seq=Seq('AGAGCGCATTTCCTACGCATGCTCGATCAAATGCTACGGATTCTAAAA'), id='seg1', name='seg1'),
           SeqRecord(seq=Seq('CCCGGACGTTTGCTCGAGGG'), id='seg2', name='seg2')]
encoding = SeqGroups(genome, sense='+', rcomp='both')
df = pd.read_hdf(os.path.join(data_path, 'test_pwms.hdf5'), key='data')
model = PWMModel(df)


def test_enrich_in_genome():
    enrichment_results = enrich_in_genome(genome, model, 10, 2, 'both', '+', 0.65, background='binomial')
    sites = enrichment_results['real_sites']
    expected = pd.read_csv('tests/data/real_sites.csv')
    assert np.all(sites.start == expected.start - 1)
    assert np.all(sites.sense == expected.sense)
    counts = enrichment_results['real_counts']
    expected = pd.read_csv('tests/data/real_counts.csv')
    assert np.all(counts.Matrix_id == expected.Matrix_id)
    assert np.all(counts.sense == expected.sense)
    assert np.all(counts.num == expected.num)
    stats = enrichment_results['shuf_stats']
    expected = pd.read_csv('tests/data/shuf_stats.csv')
    assert np.all(stats.Matrix_id == expected.Matrix_id)
    assert np.all(stats.sense == expected.sense)
    assert np.all(stats.avg == expected.avg)
    assert np.all([equals(stats.sd[i], expected.sd[i]) for i in range(len(stats))])
    enr = enrichment_results['enrichment']
    expected = pd.read_csv('tests/data/enr.csv')
    assert np.all(enr.Matrix_id == expected.Matrix_id)
    assert np.all(enr.sense == expected.sense)
    assert np.all([equals(enr.p[i], expected.p[i]) for i in range(len(enr))])
    assert np.all([equals(enr.fdr[i], expected.fdr[i]) for i in range(len(enr))])
    assert np.all(enr.adj_len == expected.adj_len)