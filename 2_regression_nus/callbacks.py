'''
Copyright (c) 2019. IIP Lab, Wuhan University
'''

import os

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import models
from tensorflow.keras import callbacks

from data import *
from layers import ProductOfExpertGaussian as POE


def _name_var_dict():
    name_var_dict = {
        "lr"          : "self.model.optimizer.lr",
        "kl_gauss"    : "self.model.sampler.gauss_loss.lamb_kl",
    }
    return name_var_dict


class AnnealEveryEpoch(callbacks.Callback):
    '''
        Anneal parameters according to some fixted
        schedule every time an epoch begins
    '''
    def __init__(self, name_schedule_dict, **kwargs):
        super(AnnealEveryEpoch, self).__init__(**kwargs)
        self.name_schedule_dict = name_schedule_dict

    def on_train_begin(self, epoch, logs=None):
        name_var_dict = _name_var_dict()
        self.var_schedule_dict = {
            name_var_dict[name]:schedule
                for name, schedule in self.name_schedule_dict.items()
        }

    def on_epoch_begin(self, epoch, logs=None):
        for var, schedule in self.var_schedule_dict.items():
            K.set_value(eval(var), schedule.value(epoch))

    def on_epoch_end(self, epoch, logs=None):
        print(), print("|"+"-"*13+"|"+"-"*10+"|")    
        for var, _ in self.var_schedule_dict.items():
            print("|{:^13}|{:^10.5f}|".format(
                eval(var).name, K.get_value(eval(var))
            ))    
        print("|"+"-"*13+"|"+"-"*10+"|"), print()


class ValidateRecordandSaveBest(callbacks.Callback):
    '''
        Evaluate model performance on validation set,
        record the training dynamic every epoch and 
        save the best model with lowest nMSE.
    '''
    def __init__(self, val_gen, rec_path, model_path, **kwargs):
        super(ValidateRecordandSaveBest, self).__init__(**kwargs)
        self.val_gen = val_gen
        self.rec_path = rec_path
        self.model_path = model_path
        self.best_nmse = np.inf

    def _build_test_model(self):
        mods_in = self.model.inputs

        encoders  = self.model.encoders
        mean_stds = [encoder(i) for encoder, i in zip(encoders, mods_in)]
        
        ### In validation, use the mode deterministically
        mean, _ = POE()(mean_stds)
        pop_level = self.model.decoder(mean)
        pred_model = models.Model(inputs=mods_in, outputs=pop_level)
        return pred_model

    def _nmse(self, preds, truth):
        return np.mean(np.square(preds - truth)) / (truth.std()**2)

    def on_train_begin(self, epoch, logs=None):
        with open(self.rec_path, "a") as f:
            f.write("nmse\n")

    def on_epoch_end(self, epoch, logs=None):
        pred_model = self._build_test_model()
        num_videos = self.val_gen.num_videos
        batch_size = self.val_gen.batch_size

        preds = np.empty([num_videos, 1], dtype=np.float32)
        truth = np.empty([num_videos, 1], dtype=np.float32)

        for i, [features, targets] in enumerate(self.val_gen):
            preds_batch = pred_model.predict(features)
            preds[i*batch_size:(i+1)*batch_size] = preds_batch
            truth[i*batch_size:(i+1)*batch_size] = targets

        ### Recover the real popularity level
        mean, std = self.val_gen.target_stats
        preds = np.exp(preds*std+mean)
        truth = np.exp(truth*std+mean)

        nmse = self._nmse(preds, truth)

        with open(self.rec_path, "a") as f:
            ### Record the training dynamic
            f.write("{}\n".format(nmse))

        if nmse < self.best_nmse:
            ### Save the best model
            self.best_nmse = nmse
            self.model.save(self.model_path)

        ### Print out the current validation metrics
        print("-"*10+"validation"+"-"*10)
        print(self.rec_path)
        print("curr nmse: {}".format(nmse))
        print("best nmse: {}".format(self.best_nmse))
        print("-"*8+"validation End"+"-"*8)


if __name__ == "__main__":
    '''
        For test purpose ONLY
    '''
    pass