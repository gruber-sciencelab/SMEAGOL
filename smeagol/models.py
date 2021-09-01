import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Input, Concatenate, Embedding, Reshape, ZeroPadding1D
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
        self.widths = np.array(pwm_df.weights.apply(lambda x:x.shape[0]))
        self.max_scores = np.array(pwm_df.weights.apply(lambda x:np.max(x, axis=1).sum()))
        self.channels = len(pwm_df)
        self.max_width = max(self.widths)
        self.define_model()
        self.set_model_weights(pwm_df.weights)
    def define_model(self):
        """Define the conv model
        """
        input_seq = Input(shape=(None, 1))
        # the padding layer will pad the sequence with zeros
        padding_layer = ZeroPadding1D(padding=(0, self.max_width - 1))
        # the one_hot_layer will one-hot-encode the sequence
        one_hot_layer = Embedding(input_dim=16, output_dim=4, input_length=None)
        reshape_layer_1 = Reshape((-1, 4))
        # the conv_layer will scan the sequence with PWMs
        conv_layer_1 = Conv1D(self.channels, [self.max_width], padding="valid", use_bias=False)
        # Connect the layers
        padded = padding_layer(input_seq)
        one_hot = one_hot_layer(padded)
        reshaped = reshape_layer_1(one_hot)
        output = conv_layer_1(reshaped)
        self.model = Model(inputs=input_seq, outputs=output, name="pwm_model")
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
        self.model.layers[2].set_weights([np.array(list(one_hot_dict.values()))])
        self.model.layers[4].set_weights([padded_weights])
    def predict(self, encoded_seqs):
        """Scan encoded sequences with the model.
        """
        predictions = self.model.predict(tf.convert_to_tensor(encoded_seqs, dtype=tf.float32))
        return predictions
    def predict_batch_with_threshold(self, encoded_seqs, thresholds, score):
        """Scan encoded sequences with the model and threshold the returned values."""
        # Inference using convolutional model
        predictions = self.predict(encoded_seqs)
        # Threshold predictions
        thresholded = np.where(predictions > thresholds)
        # Trim sites that extend beyond sequence end
        select = thresholded[1] + self.widths[thresholded[2]] <= encoded_seqs.shape[1]
        thresholded = tuple(x[select] for x in thresholded)
        # Combine site locations with scores
        if score:
            scores = predictions[thresholded]
        else:
            scores = None
        return thresholded, scores
    def predict_with_threshold(self, encoded_seqs, frac_threshold, score=False, seq_batch=0):
        """Scan encoded sequences in batches with the model and threshold the returned values.
    
        Args:
            encoded_seq (list): integer encoded sequence
            frac_threshold (float): fraction of maximum score to use as binding site detection threshold
            score (bool): Output binding site scores as well as positions
            seq_batch (int): number of sequences to scan at a time. If 0, scan all.
        
        Returns:
            thresholded (np.array): positions where the input sequence(s) match the input 
                                PWM(s) with a score above the specified threshold.
            scores (np.array): score for each potential binding site
        
        """
        # Get threshold for each PWM
        assert (frac_threshold >= 0) & (frac_threshold <= 1) 
        thresholds = frac_threshold * self.max_scores
        n_seqs = encoded_seqs.shape[0]
        if (n_seqs > 1) & (seq_batch > 0) & (seq_batch < n_seqs):
            encoded_seqs = np.vsplit(encoded_seqs, range(0, n_seqs, seq_batch)[1:])
            thresholded_list = []
            scores_list = []
            for i, batch in enumerate(encoded_seqs):
                thresholded, scores = self.predict_batch_with_threshold(batch, thresholds, score)
                thresholded = (thresholded[0] + (i*seq_batch),
                               thresholded[1],
                               thresholded[2])
                thresholded_list.append(thresholded)
                if score:
                    scores_list.append(scores)
            thresholded = tuple([np.concatenate([x[i] for x in thresholded_list]) for i in range(3)])
            del thresholded_list
            if score:
                scores = np.concatenate(scores_list)
                del scores_list
            else:
                scores = None
        else:
            thresholded, scores = self.predict_batch_with_threshold(encoded_seqs, thresholds, score)
        return thresholded, scores
            
            
        