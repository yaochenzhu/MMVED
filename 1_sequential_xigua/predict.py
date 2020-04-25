'''
Copyright (c) 2020. IIP Lab, Wuhan University
'''

import os
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import backend as K

from data import *
from train import *
from layers import ProductOfExpertGaussian as POE

### Import from train.py
train_mods = modalities

### Modality to their short name
mod_rep_dict = {
    "resnet50" :  "V",
    "audiovgg" :  "A",
    "fudannlp" :  "T",
    "social"   :  "S",
}

### Short name of the modalities
rep_mod_dict = \
    {value:key for key, value in mod_rep_dict.items()}


### Modality to their shape
mod_shape_dict = {
    "resnet50" :  128,
    "audiovgg" :  128,
    "fudannlp" :  20,
    "social"   :  3,
}


### Modality to their position in trained MMVED model
mod_pos_dict = \
    {mod:train_mods.index(mod) for mod in mod_rep_dict.keys()}


def ord_rep(rep_str):
    ord_rep = ""
    for i, letter in enumerate(["V", "A", "T", "S"]):
        if letter in rep_str:
            ord_rep += letter
    return ord_rep


def rep2mods(rep_str):
    test_mods = []
    for i, letter in enumerate(["V", "A", "T", "S"]):
        if letter in rep_str:
            test_mods.append(rep_mod_dict[letter])
    return test_mods


def mods2index(mods_list):
    idx_list = [mod_pos_dict[mod] for mod in mods_list]
    return sorted(idx_list)    


def get_model_info(model_path):
    info_dict = {}
    path_list = model_path.split(os.path.sep)
    info_dict["length"] = int(path_list[-5].split("_")[-1])
    info_dict["split"]  = int(path_list[-4])
    info_dict["lambda"] = float(path_list[-3])
    return info_dict


def get_testgen(feature_root, target_root, split_root, test_mods):
    '''
        Get data generator for test
    '''
    test_gen = VariationalEncoderDecoderGen(
        phase         = "test",        
        feature_root  = feature_root,
        target_root   = target_root,
        split_root    = split_root,
        modalities    = test_mods,
        batch_size    = 128,
        shuffle       = False, ### You can not shuffle data in test phase
        concat        = False,        
    )
    return test_gen


def build_test_model(model_path,
                     train_shapes,
                     test_mods,
                     rnn_type, 
                     summary=True):
    model = get_model(train_shapes, rnn_type, summary=False)
    model.load_weights(model_path)

    ### Get index for each modality
    mod_idxes = mods2index(test_mods)

    ### Get the input tensor indicated by mod_idxes
    mods_in = [model.inputs[:-1][i] for i in mod_idxes]
    abst_in = model.inputs[-1]

    ### Build the model for prediction
    encoders  = [model.encoders[i] for i in mod_idxes]
    mean_stds = [encoder(mod_in) for encoder, mod_in in zip(encoders, mods_in)]
    mean, _ = POE()(mean_stds)
    preds_seq = model.decoder([mean, abst_in])
    test_model = models.Model(inputs=mods_in+[abst_in], outputs=preds_seq)
    
    if summary:
        test_model.summary()
    return test_model


def predict(test_model, test_gen, save_path):
    print(test_gen.mod_shape_dict)
    num_videos = test_gen.num_videos
    batch_size = test_gen.batch_size
    timesteps  = test_gen.timesteps

    preds = np.empty([num_videos, timesteps], dtype=np.float32)
    truth = np.empty([num_videos, timesteps], dtype=np.float32)

    for i, [features, targets] in enumerate(test_gen):
        preds[i*batch_size:(i+1)*batch_size] = test_model.predict(features).squeeze()
        truth[i*batch_size:(i+1)*batch_size] = targets.squeeze()

    if save_path is not None:
        print("Prediction saved to {}".format(save_path))
        np.save(save_path, preds)
    return preds, truth


def evaluate(preds, truth, save_path):
    def pearson_corr(preds, truth):
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

    def nmse(preds, truth):   
        return np.mean(np.square(preds - truth)) / (truth.std()**2)

    corr = pearson_corr(preds, truth)
    nmse = nmse(preds, truth)

    table = pd.DataFrame({
            "nmse" : [nmse],
            "corr" : [corr] })
    print("test nmse: {:.4f}".format(nmse))
    print("test corr: {:.4f}".format(corr))
    table.to_csv(save_path, index=False, sep="\t")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
            help="path where the pretrained model is stored.")
    parser.add_argument("--data_root", type=str, default="data",
            help="path where the testing data is stored")
    parser.add_argument("--rnn_type", type=str, default="simple",
            help="path where the testing data is stored")        
    parser.add_argument("--device", type=str, default="0",
            help="specify the GPU device")
    parser.add_argument("--test_mods", type=str, default="VATS",
            help="modalities available in the test phase")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ### Set tensorflow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)    

    ### Save path to the prediction result
    model_info = get_model_info(args.model_path)
    model_root = os.path.split(args.model_path)[0]
    pred_root = os.path.join(model_root, "test", ord_rep(args.test_mods))
    if not os.path.exists(pred_root):
        os.makedirs(pred_root)
    pred_path = os.path.join(pred_root, "predict.npy")

    ### Get the test data generator
    test_mods = rep2mods(ord_rep(args.test_mods))
    target_root = os.path.join(args.data_root, "len_{}".format(model_info["length"]))
    split_root  = os.path.join(args.data_root, "split", str(model_info["split"]))
    test_gen = get_testgen(args.data_root, target_root, split_root, test_mods)

    ### Get the model for prediction
    train_shapes = [[mod_shape_dict[mod]] for mod in train_mods] + [[model_info["length"], 1]]
    test_model = build_test_model(args.model_path, train_shapes, test_mods, args.rnn_type)
    preds, truth = predict(test_model, test_gen, pred_path)

    ### Evaluate model with the nmse and corr metrics  
    eval_path = os.path.join(pred_root, "eval.txt")
    evaluate(preds, truth, eval_path)


if __name__ == '__main__':
    main()
