'''
Copyright (c) 2019. IIP Lab, Wuhan University
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


def mod2index(mod_str):
    idx_list = []
    for i, mod in enumerate(["V", "A", "T", "S"]):
        if mod in mod_str:
            idx_list.append(i)
    return idx_list    


def get_model_info(model_path):
    info_dict = {}
    path_list = model_path.split(os.path.sep)
    info_dict["split"] = int(path_list[-5])
    info_dict["lambda"] = float(path_list[-3])
    info_dict["train_mods"] = unfold_mods(path_list[-4])
    return info_dict


def get_testgen(feature_root, pred_mods, split_root):
    '''
        Get data generator for test
    '''
    test_gen = VariationalEncoderDecoderGen(
        feature_root = feature_root,
        modalities = pred_mods,
        split_root = split_root,
        phase = "test",
        batch_size  = 512,
        shuffle = False, ### you can not shuffle data in test phase
    )
    return test_gen


def build_predict_model(model_path,
                        input_shapes,
                        train_mods,
                        pred_mods,
                        summary=True):
    model = get_model(input_shapes, train_mods, summary=False)
    model.load_weights(model_path)

    ### Get index for each modality
    pred_mod_idxes = mod2index(pred_mods)

    ### Get the input tensor indicated by mod_idxes
    mods_in = [model.inputs[i] for i in pred_mod_idxes]

    ### Build the model for prediction
    encoders  = [model.encoders[i] for i in pred_mod_idxes]
    mean_stds = [encoder(mod_in) for encoder, mod_in in zip(encoders, mods_in)]
    mean, _ = POE()(mean_stds)
    preds = model.decoder(mean)
    pred_model = models.Model(inputs=mods_in, outputs=preds)

    if summary:
        pred_model.summary()
    return pred_model


def predict(model, test_gen, mod_idxes, save_path):
    num_videos = test_gen.num_videos
    batch_size = test_gen.batch_size

    preds = np.empty(num_videos, dtype=np.float32)
    truth = np.empty(num_videos, dtype=np.float32)

    for i, [feature, target] in enumerate(test_gen):
        feature = [feature[j] for j in mod_idxes]
        preds_batch = np.squeeze(model.predict(feature))
        preds[i*batch_size:(i+1)*batch_size] = preds_batch.squeeze()
        truth[i*batch_size:(i+1)*batch_size] = target.squeeze()

    mean, std = test_gen.target_stats
    preds = np.exp(preds*std+mean)
    truth = np.exp(truth*std+mean)

    if save_path is not None:
        print("Prediction saved to {}".format(save_path))
        np.save(save_path, preds)
    return preds, truth


def evaluate(preds, truth, save_path):
    def nmse(preds, truth):
        return np.mean(np.square(preds - truth)) / (truth.std()**2)

    nmse = nmse(preds, truth)
    table = pd.DataFrame({
            "nmse" : [nmse]})
    print("test nmse: {}".format(nmse))
    table.to_csv(save_path, index=False, sep="\t")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
            help="path where the pretrained model is stored.")
    parser.add_argument("--device", type=str, default="0",
            help="specify the GPU device")
    parser.add_argument("--test_mods", type=str, default="VATS",
            help="modalities available in the test phase")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    assert args.test_mods == "VATS", "Not implemented!"

    ### Set tensorflow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess  = tf.Session(config=config)
    K.set_session(sess)    

    ### Save path to the prediction result
    model_info = get_model_info(args.model_path)
    model_root = os.path.split(args.model_path)[0]
    test_root = os.path.join(model_root, "test", std_mods(args.test_mods))
    if not os.path.exists(test_root):
        os.makedirs(test_root)
    pred_path = os.path.join(test_root, "predict.npy")

    ### Get the test data generator
    feature_root = os.path.join("dataset")
    split_root = os.path.join(feature_root, "split", str(model_info["split"]))
    test_gen = get_testgen(feature_root, unfold_mods(args.test_mods), split_root)

    ### Get the model for prediction
    input_shapes = [[test_gen.mod_shape_dict[mod]] for mod in unfold_mods(args.test_mods)]
    pred_model = build_predict_model(args.model_path, input_shapes, 
                                     model_info["train_mods"], args.test_mods)
    preds, truth = predict(pred_model, test_gen, mod2index(args.test_mods), pred_path)

    ### Evaluate model with numerous indexes
    eval_path = os.path.join(test_root, "eval.txt")
    evaluate(preds, truth, eval_path)


if __name__ == '__main__':
    main()