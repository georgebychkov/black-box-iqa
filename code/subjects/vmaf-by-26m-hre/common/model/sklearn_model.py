#!/usr/bin/env python

from sklearn.svm import SVR
import numpy as np


class SklearnModel:
    def __init__(self):
        self.model = SVR()
    def fit(self, train_video_dataset, test_video_dataset):
        class SklearnDataset:
            def __init__(self, video_dataset):
                self.X = []
                self.y = []
                for X, y, _ in video_dataset.data:
                    for i in range(len(X)):
                        self.X.append((X[i]))
                        self.y.append(y[i])
        train_dataset = SklearnDataset(train_video_dataset)
        X = np.array(train_dataset.X)
        print(X.shape)
        y = np.array(train_dataset.y)
        print(y.shape)
        self.model.fit(X, y)
    def predict_on_frames(self, scaled_data):
        return self.model.predict(scaled_data)
