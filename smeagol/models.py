import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Input, Concatenate, Embedding, Reshape
from .encode import one_hot_dict

# Define convolutional model

class PWMModel:
    """Class to contain convolutional model.
    
    Args:
        pwm_df (pd.DataFrame): dataframe containing PWM IDs and weights.
    """
    def __init__(self, pwm_df):
        # Store information
        self.Matrix_ids = np.array(pwm_df.Matrix_id)
        self.widths = np.array(pwm_df.weight.apply(lambda x:x.shape[0]))
        self.max_scores = np.array(pwm_df.weight.apply(lambda x:np.max(x, axis=1).sum()))
        self.channels = len(pwm_df)
        self.max_width = max(self.widths)
        self.define_model()
        self.set_model_weights(pwm_df.weight)
    def define_model(self):
        """Define the conv model
        """
        # One-hot encode the sequence
        inputs = Input(shape=(None,1))
        l1 = Embedding(input_dim=16, output_dim=4, input_length=None)
        one_hot = l1(inputs)
        l2 = Reshape((-1, 4))
        reshaped = l2(one_hot)
        # Scan with convolutional filters                        
        l3 = Conv1D(self.channels, [self.max_width], padding="valid", use_bias=False)
        outputs = l3(reshaped)
        self.model = Model(inputs=inputs, outputs=outputs, name="pwm_model")
    def pad_weights(self, weights):
        """Function to pad weights with zeros
        """
        pad_widths = self.max_width - self.widths
        padding = [((0, pw), (0, 0)) for pw in pad_widths]
        padded_weights = [np.pad(w, p, 'constant', constant_values=0) for w, p in zip(weights, padding)]
        padded_weights = np.array(padded_weights)
        padded_weights = np.stack(padded_weights, axis=2)
        return padded_weights
    def set_model_weights(self, weights):
        """Function to fix the weights of the conv model
        """
        padded_weights = self.pad_weights(weights)
        self.model.layers[1].set_weights([np.array(list(one_hot_dict.values()))])
        self.model.layers[3].set_weights([padded_weights])
    def pad_sequence(self, encoded_seq):
        """Function to pad encoded input sequence
        """
        padded_seq = np.pad(encoded_seq, ((0,0), (0, self.max_width - 1), (0,0)), 'constant', constant_values=0)
        return padded_seq
    def predict(self, encoded_seq):
        padded_seq = self.pad_sequence(encoded_seq)
        padded_seq = tf.convert_to_tensor(padded_seq, dtype=tf.float32)
        return self.model.predict(padded_seq)