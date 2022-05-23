from smeagol.scan import _locate_sites, _bin_sites_by_score, _count_sites
from smeagol.encode import SeqEncoding
from smeagol.models import PWMModel
from smeagol.utils import _equals
import os
import pandas as pd
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


script_dir = os.path.dirname(__file__)
rel_path = "data"
data_path = os.path.join(script_dir, rel_path)

records = [SeqRecord(seq=Seq('ATTAAA'), id='S1.1', name='S1'),
           SeqRecord(seq=Seq('GCTATA'), id='S1.2', name='S1')]
encoding = SeqEncoding(records, sense='+', rcomp='none')

df = pd.read_hdf(os.path.join(data_path, 'test_pwms.hdf5'), key='data')
model = PWMModel(df)
thresholded = [[1, 1],
               [0, 0],
               [0, 2]]
scores = np.array([2.343, 1.929])


def test_locate_sites():
    result = _locate_sites(encoding, model, thresholded, scores)
    fields = ['id', 'name', 'sense', 'start', 'Matrix_id', 'width', 'end', 'score']
    expected = [['S1.2', 'S1.2'], ['S1', 'S1'], ['+', '+'], [0, 0], ['x', 'z'], [3, 3], [3, 3], [2.343, 1.929]]
    for field, exp in zip(fields, expected):
        assert np.all(result[field].values == exp)
    assert _equals(result.max_score.values, np.array([3.33376361, 2.50816981]))
    assert _equals(result.frac_score.values, np.array([0.70280927926, 0.76908668]))


def test_bin_sites_by_score():
    thresholded, scores = model.predict_with_threshold(encoding.seqs, 0.25, True)
    result = _bin_sites_by_score(encoding, model, thresholded, scores, [.25, .5, .75])
    fields = ['id', 'name', 'sense', 'Matrix_id', 'width']
    expected = [['S1.1', 'S1.1', 'S1.2', 'S1.2', 'S1.2', 'S1.2'], ['S1'] * 6, ['+'] * 6, ['x', 'z', 'x', 'z', 'x', 'z'], [3] * 6]
    for field, exp in zip(fields, expected):
        assert np.all(result[field].values == exp)
    assert np.all([str(x) for x in result.bin] == ['(0.25, 0.5]']*2 + ['(0.5, 0.75]']*2 + ['(0.75, 1.0]']*2)
    assert np.all(result.num == [0, 2, 1, 0, 0, 1])


def test_count_sites():
    result = _count_sites(encoding, model, thresholded)
    fields = ['id', 'name', 'sense', 'Matrix_id', 'width', 'num']
    expected = [['S1.2', 'S1.2'], ['S1', 'S1'], ['+', '+'], ['x', 'z'], [3, 3], [1, 1]]
    for field, exp in zip(fields, expected):
        assert np.all(result[field].values == exp)