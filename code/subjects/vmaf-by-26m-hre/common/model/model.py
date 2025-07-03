#!/usr/bin/env python

from common.extractor import load_video
from common.model.neural_model import NeuralModel
from common.model.rnn_model import RNNModel
from common.model.sklearn_model import SklearnModel

from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import pickle


class Model:
    def __init__(self, model_type):
        if model_type == "sklearn":
            self.model = SklearnModel()
        elif model_type == "neural":
            self.model = NeuralModel()
        elif model_type == "rnn":
            self.model = RNNModel()

        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, dataset, verbose=True):
        train_videos = dataset.get_train_features()
        test_videos = dataset.get_test_features()

        self.scaler.fit(np.concatenate(
            [X for X, _, _ in train_videos] + [X for X, _,  _ in test_videos]))
        for X, _, _ in train_videos:
            X[:] = self.scaler.transform(X)
        for X, _, _ in test_videos:
            X[:] = self.scaler.transform(X)
        self.model.fit(train_videos, test_videos)
        self.fitted = True

    def predict(self, ref, dist, mode="video", return_vmaf=False):
        if not self.fitted:
            raise RuntimeError("model must be fitted before usage")
        data = load_video(ref, dist)
        vmaf = data[:, -2]
        model_output = self.predict_on_data(data, mode)
        outputs = [model_output]
        if return_vmaf:
            outputs.append(vmaf if mode == "frames" else vmaf.mean())
        if len(outputs) > 1:
            return tuple(outputs)
        else:
            return outputs[0]


    def predict_on_data(self, data, mode="video"):
        frames = self.model.predict_on_frames(self.scaler.transform(data))
        return frames if mode == "frames" else frames.mean()
