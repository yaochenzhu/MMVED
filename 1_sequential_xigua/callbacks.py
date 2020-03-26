'''
Copyright (c) 2020. IIP Lab, Wuhan University
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
        save the best model with lowest nMSE or Corr.
    '''
    def __init__(self, val_gen, rec_path, model_root, **kwargs):
        super(ValidateRecordandSaveBest, self).__init__(**kwargs)
        self.val_gen = val_gen
        self.rec_path = rec_path
        self.model_root = model_root
        self.best_nmse = np.inf
        self.best_corr = -np.inf

    def _build_test_model(self):
        mods_in = self.model.inputs[:-1]
        abst_in  = self.model.inputs[-1]

        encoders  = self.model.encoders
        mean_stds = [encoder(i) for encoder, i in zip(encoders, mods_in)]

        ### In validation, use the mode deterministically
        mean, _ = POE()(mean_stds)
        pop_sequence = self.model.decoder([mean, abst_in])
        pred_model = models.Model(inputs=mods_in + [abst_in], outputs=pop_sequence)
        return pred_model

    def _pearson_corr(self, preds, truth):
        corr = 0
        num_samples = len(preds)
        cnt_samples = num_samples
        for i in range(num_samples):
            corr_this =  pd.Series(preds[i]).corr(pd.Series(truth[i]))
            if np.isnan(corr_this):
                cnt_samples = cnt_samples-1
                continue
            corr += corr_this
        return corr / cnt_samples

    def _nmse(self, preds, truth):
        return np.mean(np.square(preds - truth)) / (truth.std()**2)

    def on_train_begin(self, epoch, logs=None):
        with open(self.rec_path, "a") as f:
            f.write("nmse\tcorr\n")

    def on_epoch_end(self, epoch, logs=None):
        pred_model = self._build_test_model()
        num_videos = self.val_gen.num_videos
        batch_size = self.val_gen.batch_size
        timesteps  = self.val_gen.timesteps

        preds = np.empty([num_videos, timesteps], dtype=np.float32)
        truth = np.empty([num_videos, timesteps], dtype=np.float32)

        for i, [features, targets] in enumerate(self.val_gen):
            preds_batch   = np.squeeze(pred_model.predict(features))
            targets_batch = np.squeeze(targets)
            preds[i*batch_size:(i+1)*batch_size] = preds_batch
            truth[i*batch_size:(i+1)*batch_size] = targets_batch

        nmse = self._nmse(preds, truth)
        corr = self._pearson_corr(preds, truth)

        with open(self.rec_path, "a") as f:
            ### Record the training dynamic
            f.write("{}\t{}\n".format(nmse, corr))

        if nmse < self.best_nmse:
            ### Save the best model for nmse
            self.best_nmse = nmse
            self.model.save(os.path.join(self.model_root, "best_nmse.h5"))

        if corr > self.best_corr:
            ### Save the best model for corr
            self.best_corr = corr
            self.model.save(os.path.join(self.model_root, "best_corr.h5"))

        ### Print out the current validation metrics
        print("-"*10+"validation"+"-"*10)
        print(self.rec_path)
        print("curr nmse: {}; curr corr: {}".format(nmse, corr))
        print("best nmse: {}; best corr: {}".format(self.best_nmse, self.best_corr))
        print("-"*8+"validation End"+"-"*8)


if __name__ == "__main__":
    '''
        For test purpose ONLY
    '''
    pass