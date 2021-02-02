import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.layers import Conv1D, Input, Concatenate, Embedding, Reshape
from .encoding import one_hot_dict

# Define convolutional model

class PWMModel:
    def __init__(self, pwm_df):
        df = pwm_df.copy()
        df['width'] = df.weight.apply(lambda x:x.shape[0])
        df = df.sort_values('width').reset_index(drop=True)
        self.Matrix_ids = np.array(df.Matrix_id)
        self.widths = np.array(df.width)
        self.unique_widths = np.unique(self.widths)
        self.weights = np.array(df.weight)
        self.max_scores = np.array(df.weight.apply(lambda x:np.max(x, axis=1).sum())) 
        self.get_conv_model()
    def get_conv_model(self):
        # One-hot encode the sequence
        inputs = Input(shape=(None,1))
        e1 = Embedding(input_dim=16, output_dim=4, input_length=None)
        one_hot = e1(inputs)
        e2 = Reshape((-1, 4))
        reshaped = e2(one_hot)
        # Scan with convolutional kernels based on PWMs                        
        if len(self.unique_widths) == 1:
            w = self.unique_widths[0]
            l = Conv1D(sum(self.widths == w), [w], padding="valid", use_bias=False)
            outputs = l(reshaped)
        else:
            outputs=[]
            for w in self.unique_widths:
                l = Conv1D(sum(self.widths == w), [w], padding="valid", use_bias=False)
                outputs.append(l(reshaped))
        model = Model(inputs=inputs, outputs=outputs, name="pwm_model")
        # Fix weights
        e1.set_weights([np.array(list(one_hot_dict.values()))])
        if len(self.unique_widths) == 1:
            l = model.layers[3]
            weights = np.stack(self.weights, axis=2)
            l.set_weights([weights])
        else:
            for i in range(3, len(model.layers)):
                l = model.layers[i]
                w = self.unique_widths[i-3]
                weights = np.stack(self.weights[self.widths == w], axis=2)
                l.set_weights([weights])
        self.model = model
    def predict(self, inp):
        return self.model.predict(inp)