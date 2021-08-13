from smeagol.scan import *
from smeagol.encode import SeqEncoding
from smeagol.models import PWMModel
from smeagol.utils import equals
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


script_dir = os.path.dirname(__file__)
rel_path = "data"
data_path = os.path.join(script_dir, rel_path)

records = [SeqRecord(seq=Seq('ATTAAA'), id='S1.1', name='S1'),
           SeqRecord(seq=Seq('GCTATA'), id='S1.2', name='S1')]
encoding = SeqEncoding(records, sense='+', rcomp=None)

df = pd.read_hdf(os.path.join(data_path, 'test_pwms.hdf5'), key='data')
model = PWMModel(df)
thresholded = [(1, 1),
               (0, 0),
               (0, 2)]
scores = np.array([2.343, 1.929])




def test_locate_sites():
    result = locate_sites(encoding, model, thresholded, scores)
    assert result.id == ['S1.2', 'S1.2']
    assert result.name == ['S1', 'S1']
    assert result.sense == ['+', '+']
    assert result.start == [0, 0]
    assert result.Matrix_id == ['x', 'z']
    assert result.width == [3, 3]
    assert result.end == [3, 3]
    assert result.score == [2.343, 1.929]
    assert equals(result.max_score, [3.33376361, 2.50816981])
    assert equals(result.frac_score, [0.70280927926, 0.76908668])


def test_bin_sites_by_score():
    result = bin_sites_by_score(encoding, model, thresholded, scores, [.25, .5, .75])
    assert result.Matrix_id == ['z', 'x', 'z', 'z']
    assert result.width == [3] * 4
    assert result.id == ['S1.1', 'S1.2', 'S1.2', 'S1.2']
    assert result.name == ['S1'] * 4
    assert result.sense == ['*'] * 4
    assert [str(x) for x in result.bin] == ['(0.25, 0.5]', '(0.5, 0.75]', '(0.75, 1.0]',
                                            '(0.25, 0.5]']
    assert result.num == [2, 1, 1, 1]


def test_count_sites():
    result = count_sites(encoding, model, thresholded)
    assert result.Matrix_id == ['x', 'z']
    assert result.width == [3, 3]
    assert result.id == ['S1.2', 'S1.2']
    assert result.name == ['S1', 'S1']
    assert result.sense == ['+', '+']
    assert result.num == [1, 1]