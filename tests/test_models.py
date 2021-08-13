from smeagol.models import *
import os
import pandas as pd
from smeagol.encode import one_hot_dict
from smeagol.utils import equals


script_dir = os.path.dirname(__file__)
rel_path = "data"
data_path = os.path.join(script_dir, rel_path)


def test_PWMModel():
    df = pd.read_hdf(os.path.join(data_path, 'test_pwms.hdf5'), key='data')
    model = PWMModel(df)
    assert model.Matrix_ids == ['x', 'y', 'z']
    assert model.widths == [3, 5, 3]
    assert equals(model.max_scores, np.array([3.33376361, 5.52770673, 2.50816981]))
    assert model.channels == 3
    assert model.max_width == 5
    assert [str(type(x)) for x in model.model.layers] == ["<class 'tensorflow.python.keras.engine.input_layer.InputLayer'>",
     "<class 'tensorflow.python.keras.layers.convolutional.ZeroPadding1D'>",
     "<class 'tensorflow.python.keras.layers.embeddings.Embedding'>",
     "<class 'tensorflow.python.keras.layers.core.Reshape'>",
     "<class 'tensorflow.python.keras.layers.convolutional.Conv1D'>"]
    assert equals(np.array(model.model.layers[2].weights[0]), np.array(list(one_hot_dict.values())))
    assert model.model.layers[4].weights[0].shape == (5, 4, 3)
    assert equals(model.model.layers[4].weights[0][:, :, 0], np.pad(x, ((0, 2),(0, 0))))
    assert equals(model.model.layers[4].weights[0][:, :, 1], y)
    assert equals(model.model.layers[4].weights[0][:, :, 2], np.pad(z, ((0, 2),(0, 0))))
    preds = model.predict([1, 2, 3])
    assert preds.shape == (3, 1, 3)
    assert equals(preds[:, 0, 0], [-1.60847875, -13.31642296, -6.65821148])
    preds = model.predict_with_threshold([1,2,3], 0.2, score=True)
    assert preds[0] == (np.array([0, 0, 0]), np.array([0, 2, 2]), np.array([2, 0, 2]))
    assert equals(preds[1], np.array([1.668218, 0.35611725000000005,  1.25467785]))
