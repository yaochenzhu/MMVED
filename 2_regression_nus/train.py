'''
Copyright (c) 2019. IIP Lab, Wuhan University
'''

import os
import time
import argparse

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler

from utils import PiecewiseSchedule
from model import VariationalEncoderDecoder
from data  import VariationalEncoderDecoderGen
from callbacks import AnnealEveryEpoch, ValidateRecordandSaveBest

import warnings
warnings.filterwarnings('ignore')


def get_gen(feature_root, modalities, split_root, batch_size):
    train_gen = VariationalEncoderDecoderGen(
        phase = "train",
        feature_root = feature_root,
        modalities = modalities,
        split_root = split_root,
        batch_size = batch_size,
        shuffle = True,
    )
    val_gen = VariationalEncoderDecoderGen(
        phase = "val", 
        feature_root = feature_root,
        modalities = modalities,
        split_root = split_root,
        batch_size  = batch_size,
        shuffle = False,
    )
    return train_gen, val_gen


def get_model(input_shapes, modalities, summary=True):
    mod_args_dict = {
        "visual"  : {"hidden_sizes":[32, 8], "activation":"relu"},
        "aural"   : {"hidden_sizes":[32, 8], "activation":"relu"},
        "textual" : {"hidden_sizes":[32, 8], "activation":"relu"},
        "social"  : {"hidden_sizes":[32, 8], "activation":"relu"},
    }
    enc_args = [mod_args_dict[mod] for mod in modalities]
    dec_args = {"hidden_sizes":[4], "activation":"relu"}
    model = VariationalEncoderDecoder(enc_args, dec_args, input_shapes)
    if summary:
        model.summary()
    return model


def train(model, epochs, train_gen, val_gen, lambd, rec_path, model_path):
    lr_schedule = PiecewiseSchedule(
        [[0       , 5e-4],
         [epochs/2, 1e-5],
         [epochs  , 5e-5]], 
         outside_value=5e-4)        

    ### In our implementation, we keep lambd fixed.
    lambda_schedule = PiecewiseSchedule(
        [[0,      lambd],
         [epochs, lambd]], 
         outside_value=lambd)

    callbacks = [
        AnnealEveryEpoch({"lr": lr_schedule, "kl_gauss":lambda_schedule}),
        ValidateRecordandSaveBest(val_gen, rec_path, model_path),
    ]

    model.compile(optimizer=optimizers.Adam(), loss="mse")
    model.fit_generator(train_gen, workers=4, epochs=epochs, callbacks=callbacks)    


def std_mods(mod_str):
    mod_str = mod_str.upper()
    std_mod = ""
    for mod in ["V","A","T","S"]:
        if mod in mod_str:
            std_mod += mod
    return std_mod


def unfold_mods(mod_str):
    unfolded_mods = []
    mod_str = mod_str.upper()
    for s_mod, l_mod in zip(["V","A","T","S"],
                            ["visual", "aural", "textual", "social"]):
        if s_mod in mod_str:
            unfolded_mods.append(l_mod)
    return unfolded_mods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambd", type=float, default=0.3,
        help="weights of the KL divergence, control the wideness of information bottleneck.")
    parser.add_argument("--modalities", type=str, default="VATS",
        help="modality used for training, by defaut we use features from all four modalities.")    
    parser.add_argument("--batch_size", type=int, default=512,
        help="size of training batch, too small of which may leads to instability.")
    parser.add_argument("--epochs", type=int, default=50,
        help="number of training epochs")
    parser.add_argument("--split", type=int, default=0,
        help="use which split of train/val/test")
    parser.add_argument("--device", type=str, default="0",
        help="use which GPU device")    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    ### Set tensorflow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    ### Get the data generator
    feature_root = os.path.join("dataset")
    modalities = unfold_mods(args.modalities)
    split_root = os.path.join(feature_root, "split", str(args.split))
    train_gen, val_gen = get_gen(feature_root, modalities, split_root, args.batch_size)

    ### Get input shapes and set up the model
    input_shapes = [[train_gen.mod_shape_dict[mod]] for mod in modalities]
    model = get_model(input_shapes, modalities, "poe")

    ### Set up the training process 
    save_root = os.path.join("models", str(args.split), std_mods(args.modalities),
                             str(args.lambd), time.strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(save_root):
        os.makedirs(save_root)
        
    rec_path = os.path.join(save_root, "validation_records.txt")
    model_path = os.path.join(save_root, "best_model.h5")
    train(model, args.epochs, train_gen, val_gen, args.lambd, rec_path, model_path)


if __name__ == '__main__':
    main()