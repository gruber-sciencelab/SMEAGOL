from smeagol.models import *
import os
import pandas as pd
from smeagol.utils import _equals
from smeagol.encode import one_hot_encode

script_dir = os.path.dirname(__file__)
rel_path = "data"
data_path = os.path.join(script_dir, rel_path)

layer_classes = [
    ["<class 'tensorflow.python.keras.engine.input_layer.InputLayer'>",
     "<class 'tensorflow.python.keras.layers.convolutional.ZeroPadding1D'>",
     "<class 'tensorflow.python.keras.layers.embeddings.Embedding'>",
     "<class 'tensorflow.python.keras.layers.core.Reshape'>",
     "<class 'tensorflow.python.keras.layers.convolutional.Conv1D'>"],
    ["<class 'keras.engine.input_layer.InputLayer'>",
     "<class 'keras.layers.reshaping.zero_padding1d.ZeroPadding1D'>",
     "<class 'keras.layers.core.embedding.Embedding'>",
     "<class 'keras.layers.reshaping.reshape.Reshape'>",
     "<class 'keras.layers.convolutional.conv1d.Conv1D'>"]
]


def test_PWMModel():
    df = pd.read_hdf(os.path.join(data_path, 'test_pwms.hdf5'), key='data')
    x = df.weights[0]
    y = df.weights[1]
    z = df.weights[2]
    model = PWMModel(df)

    # Check variables
    assert np.all(model.Matrix_ids == ['x', 'y', 'z'])
    assert np.all(model.widths == [3, 5, 3])
    assert _equals(model.max_scores, np.array([3.33376361, 5.52770673, 2.50816981]))
    assert model.channels == 3
    assert model.max_width == 5

    # Check layers
    assert [str(type(x)) for x in model.model.layers] in layer_classes

    # Check weights
    assert _equals(np.array(model.model.layers[2].weights[0]), np.array(list(one_hot_dict.values())))
    assert np.all(model.model.layers[4].weights[0].shape == (5, 4, 3))
    assert _equals(np.array(model.model.layers[4].weights[0][:, :, 0]), np.pad(x, ((0, 2),(0, 0))))
    assert _equals(np.array(model.model.layers[4].weights[0][:, :, 1]), y)
    assert _equals(np.array(model.model.layers[4].weights[0][:, :, 2]), np.pad(z, ((0, 2),(0, 0))))

    # Test prediction
    encoded_seq = np.array([[1, 2, 3]])
    preds = model.predict(encoded_seq)
    assert preds.shape == (1, 3, 3)
    assert _equals(preds[0, 0, :], np.array([-1.60847875, -7.32647115, 0.35611725000000005]))
    preds = model.predict_with_threshold(encoded_seq, 0.1, score=True)
    assert preds[0] == (np.array([0]), np.array([0]), np.array([2]))
    assert _equals(preds[1], np.array([0.35611725]))
