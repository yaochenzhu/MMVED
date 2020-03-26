'''
Copyright (c) 2019. IIP Lab, Wuhan University
'''

import os
import glob

import numpy as np
import pandas as pd
from tensorflow import keras


class VariationalEncoderDecoderGen(keras.utils.Sequence):
    '''
        Generate training data, validation, test data
    '''
    def __init__(self,
                 feature_root,
                 modalities,
                 split_root,
                 phase,
                 batch_size,
                 shuffle=True):

        assert phase in ["train", "val", "test"], \
            "phase must be one of train, val, test!"

        ### Load the indexes of the videos 
        index_path = os.path.join(split_root, "{}.txt".format(phase))
        phase_idxes = pd.read_table(index_path, header=None).values.squeeze()

        ### Load the features from specified modalities
        self.modalities = modalities
        self.features = []
        for modality in modalities:
            feature_path = os.path.join(feature_root, "{}.npy".format(modality))
            self.features.append(
                np.load(feature_path)[phase_idxes]
            )

        ### Load the groundtruth
        target_path = os.path.join(feature_root, "target.npz")
        self.target = np.load(target_path)["target"][phase_idxes]
        self.target_stats = [np.load(target_path)["mean"], np.load(target_path)["std"]]

        ### Data&Batch information
        self.num_videos  = len(phase_idxes)
        self.video_idxes = np.arange(self.num_videos)
        self.batch_size  = batch_size

        ### Shuffle the video indexes if necessary
        self.shuffle = shuffle
        if self.shuffle:
            self.on_epoch_end()

    def on_epoch_end(self): 
        '''
            Shuffle the index after each epoch finished
        '''
        if self.shuffle:
            np.random.shuffle(self.video_idxes)    

    def __len__(self): 
        '''
            The total number of batches
        '''
        self.batch_num = self.num_videos // self.batch_size + 1
        return self.batch_num

    def __getitem__(self, i):
        '''
            Return the batch indexed by i
        '''
        batch_idxes = self.video_idxes[i*self.batch_size:(i+1)*self.batch_size]
        batch_size  = len(batch_idxes)

        batch_features = [np.empty((batch_size, feature.shape[-1]), dtype=np.float32) 
                                for feature in self.features]
        batch_target = np.empty((batch_size, 1), dtype=np.float32)

        for j, idx in enumerate(batch_idxes):
            for k in range(len(self.modalities)):
                batch_features[k][j] = self.features[k][idx]
            batch_target[j] = self.target[idx]

        return batch_features, batch_target

    @property
    def mod_shape_dict(self):
        return {mod:feature.shape[-1] for mod, feature in zip(self.modalities, self.features)}


if __name__ == "__main__":
    '''
        For test purpose ONLY
    '''
    pass