from smeagol.variant import *
import pandas as pd
import numpy as np


script_dir = os.path.dirname(__file__)
rel_path = "data"
data_path = os.path.join(script_dir, rel_path)

records = [SeqRecord(seq=Seq('ATTAAA'), id='seq0', name='seq0'),
           SeqRecord(seq=Seq('GCTATA'), id='seq1', name='seq1')]
encoding = SeqEncoding(records, sense='+', rcomp=None)

df = pd.read_hdf(os.path.join(data_path, 'test_pwms.hdf5'), key='data')


sites = pd.DataFrame({'name':['seq0', 'seq1'], 
                      'start':[1, 10], 
                      'Matrix_id':[pwm, pwm],
                      'width':[3, 3],
                      'end': [4, 13]})

variants = pd.DataFrame({'name':['seq0', 'seq0', 'seq1', 'seq1'], 
                         'pos':[2, 5, 11, 12], 
                         'alt':['C', 'T', 'A', 'A']})
