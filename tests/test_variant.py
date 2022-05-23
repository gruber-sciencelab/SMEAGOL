from smeagol.variant import *
from smeagol.encode import SeqEncoding
from smeagol.utils import _equals

import pandas as pd
import numpy as np
import os

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord



script_dir = os.path.dirname(__file__)
rel_path = "data"
data_path = os.path.join(script_dir, rel_path)


records = [SeqRecord(seq=Seq('GACAAA'), id='seq0', name='seq0'),
           SeqRecord(seq=Seq('GCTATA'), id='seq1', name='seq1')]
encoding = SeqEncoding(records, sense='+', rcomp='none')


df = pd.read_hdf(os.path.join(data_path, 'test_pwms.hdf5'), key='data')


sites = pd.DataFrame({'name':['seq0', 'seq1'], 
                      'start':[0, 0], 
                      'Matrix_id':['x', 'z'],
                      'width':[3, 3],
                      'end': [3, 3]})

variants = pd.DataFrame({'name':['seq0', 'seq0', 'seq1', 'seq1'], 
                         'pos':[2, 5, 0, 1], 
                         'alt':['G', 'T', 'A', 'A']})



def test_variant_effect_on_sites():
    result = variant_effect_on_sites(sites, variants, records, df)
    assert len(result) == 3
    assert np.all(result['name'].values == ['seq0', 'seq1', 'seq1'])
    assert np.all(result['pos'].values == [2, 0, 1])
    assert np.all(result['alt'].values == ['G', 'A', 'A'])
    assert np.all(result['start'].values == [0, 0, 0])
    assert np.all(result['end'].values == [3, 3, 3])
    assert np.all(result['Matrix_id'].values == ['x', 'z', 'z'])
    assert np.all(result['seq'].values == ['GAC', 'GCT', 'GCT'])
    assert _equals(result.max_score.values, np.array([3.33376361, 2.50816981, 2.50816981]))
    assert _equals(result.score.values, np.array([1., 0.769149, 0.769149]))
    assert _equals(result.variant_score.values, np.array([0.4080668, 0.1419828, 0.769149]))
