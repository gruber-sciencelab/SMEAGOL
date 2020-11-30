import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.layers import Conv1D, Input, Concatenate

# Define convolutional model

class PWMModel:
    def __init__(self, pwm_df):
        df = pwm_df.copy()
        df['width'] = df.weight.apply(lambda x:x.shape[0])
        df = df.sort_values('width').reset_index(drop=True)
        self.Matrix_ids = np.array(df.Matrix_id)
        self.widths = np.array(df.width)
        self.weights = np.array(df.weight)
        self.max_scores = np.array(df.weight.apply(lambda x:np.max(x, axis=1).sum())) 
        self.get_conv_model()
    def get_conv_model(self):
        unique_widths = np.unique(self.widths)
        inputs = Input(shape=(None,4))
        if len(unique_widths) == 1:
            w = unique_widths[0]
            l = Conv1D(sum(self.widths == w), [w], padding="same", use_bias=False)
            outputs = l(inputs)
        else:
            pwm_outputs=[]
            for w in unique_widths:
                l = Conv1D(sum(self.widths == w), [w], padding="same", use_bias=False)
                pwm_outputs.append(l(inputs))
            conc = Concatenate(axis=2)
            outputs = conc(pwm_outputs)
        model = Model(inputs=inputs, outputs=outputs, name="pwm_model")
        # Fix weights
        for i in range(1, len(model.layers)-1):
            l = model.layers[i]
            w = unique_widths[i-1]
            weights = np.stack(self.weights[self.widths == w], axis=2)
            l.set_weights([weights])
        self.model = model
    def predict(self, inp):
        return self.model.predict(inp)