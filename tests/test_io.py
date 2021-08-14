from smeagol.io import *
import os

script_dir = os.path.dirname(__file__)
rel_path = "data"
data_path = os.path.join(script_dir, rel_path)


def test_read_fasta():
    input_file = os.path.join(data_path, 'test.fa.gz')
    records = read_fasta(input_file)
    expected = [SeqRecord(seq=Seq('ATTAAATA'), id='Seg1', name='Seg1', 
                          description='Test sequence, segment 1'), 
                SeqRecord(seq=Seq('CAAAATCTTTAGGATTAGCAC'), id='Seg2', name='Seg2', 
                          description='Test sequence, segment 2')]
    assert [record.seq for record in records] == [record.seq for record in expected]
    assert [record.id for record in records] == [record.id for record in expected]
    assert [record.name for record in records] == [record.name for record in expected]


def test_write_fasta():
    records = [SeqRecord(seq=Seq('ATTAAATA'), id='Seg1', name='Seg1', 
                          description='Test sequence, segment 1'), 
                SeqRecord(seq=Seq('CAAAAT'), id='Seg2', name='Seg2', 
                          description='Test sequence, segment 2')]
    write_fasta(records, 'test_write_fasta.fa.gz', gz=True)
    result = [x.decode() for x in gzip.open('test_write_fasta.fa.gz')]
    expected = ['>Seg1 Test sequence, segment 1\n', 'ATTAAATA\n',
                '>Seg2 Test sequence, segment 2\n', 'CAAAAT\n']
    assert np.all(result == expected)
    os.remove('test_write_fasta.fa.gz')


def test_read_pms_from_file():
    input_file = os.path.join(data_path, 'test_pwm_file.txt')
    result = read_pms_from_file(input_file, matrix_type='PPM',
                                check_lens=False, transpose=False, delimiter='\t')
    expected = pd.DataFrame({'Matrix_id':['Matrix0', 'Matrix1'], 
                             'probs':[np.array([[0.97, 0.01, 0.01, 0.01], 
                                                [0.97, 0.01, 0.01, 0.01], 
                                                [0.4, 0.2, 0.01, 0.39], 
                                                [0.97, 0.01, 0.01, 0.01]], dtype='float32'), 
                                     np.array([[0.97, 0.01, 0.01, 0.01], 
                                               [0.58, 0.01, 0.4, 0.01], 
                                               [0.01, 0.2, 0.4, 0.39]], dtype='float32')]})
    assert len(result) == 2
    assert np.all(result.Matrix_id == expected.Matrix_id)
    assert np.all(result.probs[0] == expected.probs[0])
    assert np.all(result.probs[1] == expected.probs[1])

    
def test_read_pms_from_dir():
    input_dir = os.path.join(data_path, 'test_PFMdir')
    result = read_pms_from_dir(input_dir, matrix_type='PFM', transpose=True)
    assert len(result) == 2
    assert np.all(result.Matrix_id.values == ['Matrix0', 'Matrix1'])
    assert np.all(result.freqs[0] == np.array([[2, 18, 0, 0],
                                                [2, 16, 1, 1],
                                                [0, 2, 18, 0]], dtype='float32'))
    assert np.all(result.freqs[1] == np.array([[0, 0, 1, 19],
                                               [20, 0, 0, 0],
                                               [1, 4, 13, 2],
                                               [0, 1, 17, 2]], dtype='float32'))