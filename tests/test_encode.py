from smeagol.encode import *
import os
import pytest

script_dir = os.path.dirname(__file__)
rel_path = "data"
data_path = os.path.join(script_dir, rel_path)


def test_integer_encode():
    record = SeqRecord(Seq('ACGTNWSMKRYBDHVZ'), id='id', name='name')
    result = integer_encode(record, rcomp=False)
    assert result == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]
    result = integer_encode(record, rcomp=True)
    assert result == [0,12,13,14,15,10,11,8,9,7,6,5,1,2,3,4]
    record = SeqRecord(Seq('ACGE'))
    with pytest.raises(KeyError):
        integer_encode(record, rcomp=False)
 
    
def test_SeqEncoding():
    records = [SeqRecord(Seq('AGC'), id='id1', name='name1'),
               SeqRecord(Seq('ACA'), id='id2', name='name2')]
    result = SeqEncoding(records, sense='+')
    assert result.len == 3
    assert np.all(result.ids == ['id1', 'id2'])
    assert np.all(result.names == ['name1', 'name2'])
    assert np.all(result.seqs == np.array([[1,3,2], [1,2,1]]))
    assert np.all(result.senses == ['+', '+'])
    assert type(result.senses) == np.ndarray
    assert type(result.ids) == np.ndarray
    assert type(result.names) == np.ndarray
    assert type(result.senses) == np.ndarray
    result = SeqEncoding(records, sense='+', rcomp='only')
    assert np.all(result.ids == ['id1', 'id2'])
    assert np.all(result.names == ['name1', 'name2'])
    assert np.all(result.seqs == np.array([[3,2,4], [4,3,4]]))
    assert np.all(result.senses == ['-', '-'])
    result = SeqEncoding(records, sense='+', rcomp='both')
    assert np.all(result.ids == ['id1', 'id2', 'id1', 'id2'])
    assert np.all(result.names == ['name1', 'name2', 'name1', 'name2'])
    assert np.all(result.seqs == np.array([[1,3,2], [1,2,1], [3,2,4], [4,3,4]]))
    assert np.all(result.senses == ['+', '+', '-', '-'])

    
def test_SeqGroups():
    records = os.path.join(data_path, 'test.fa.gz')
    result = SeqGroups(records, sense='+')
    expected = [SeqEncoding([SeqRecord(seq=Seq('ATTAAATA'), id='Seg1', name='Seg1')], sense='+', rcomp=None), SeqEncoding([SeqRecord(seq=Seq('CAAAATCTTTAGGATTAGCAC'), id='Seg2', name='Seg2')], sense='+', rcomp=None)]
    for i in range(2):
        assert np.all(result.seqs[i].seqs == expected[i].seqs)
        assert result.seqs[i].ids == expected[i].ids
        assert result.seqs[i].senses == expected[i].senses
        assert result.seqs[i].names == expected[i].names
    records = [SeqRecord(Seq('AGC'), id='id1', name='name'),
               SeqRecord(Seq('ACA'), id='id2', name='name')]
    result = SeqGroups(records, sense='-', rcomp='both', group_by='name')
    assert len(result.seqs) == 1
    expected = SeqEncoding(records, sense='-', rcomp='both')
    assert np.all(result.seqs[0].seqs == expected.seqs)
    assert np.all(result.seqs[0].ids == expected.ids)
    assert np.all(result.seqs[0].names == expected.names)
    assert np.all(result.seqs[0].senses == expected.senses)
    