import os, re 
import numpy as np
from sklearn.model_selection import train_test_split

class AudioDataset(object):
    def __init__(self, train_data_path, labels_path, batch_size):
        self.data = np.load(train_data_path)
        self.labels = np.load(labels_path)
        self.batch_size = batch_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, test_size=0)
        self.train_num = self.X_train.shape[0]
        self.test_num = self.X_test.shape[0]
        
        self.train_batch_count = int(self.train_num/self.batch_size) - 1 if self.train_num % self.batch_size == 0 \
            else int(self.train_num/self.batch_size)
        self.test_batch_count = int(self.test_num/self.batch_size) - 1 if self.test_num % self.batch_size == 0 \
            else int(self.test_num/self.batch_size)
    
    def get_batch(self, index, subset="train"):
        if subset == "train":
            x, y = self.X_train, self.y_train
            max_index = self.train_batch_count
        else:
            x, y = self.X_test, self.y_test
            max_index = self.test_batch_count
        if index > max_index:
            raise ValueError("index out of range, maximum index: %d" % max_index)
        elif (index == max_index) and (x.shape[0] % self.batch_size != 0):
            choice = np.random.choice(range(x.shape[0]), self.batch_size-(x.shape[0]%self.batch_size))
            x_batch = x[index*self.batch_size:x.shape[0]] + [x[idx] for idx in choice]
            y_batch = y[index*self.batch_size:y.shape[0]] + [y[idx] for idx in choice]
        else:
            x_batch = x[index*self.batch_size:(index+1)*self.batch_size]
            y_batch = y[index*self.batch_size:(index+1)*self.batch_size]
        return {"X_train":x_batch, "y_train":y_batch}

    def shuffle(self):
        if subset == "train" or subset == "train_and_valid":
            index = list(range(self.train_doc_count))
            np.random.shuffle(index)
            self.train_docs = [self.train_docs[idx] for idx in index]
            self.train_chars = [self.train_chars[idx] for idx in index]
            self.train_taggings = [self.train_taggings[idx] for idx in index]
            self.train_texts = [self.train_texts[idx] for idx in index]
            self.train_mask = [self.train_mask[idx] for idx in index]
            if subset == "train_and_valid":
                index = list(range(self.valid_doc_count))
                np.random.shuffle(index)
                self.valid_docs = [self.valid_docs[idx] for idx in index]
                self.valid_chars = [self.valid_chars[idx] for idx in index]
                self.valid_taggings = [self.valid_taggings[idx] for idx in index]
                self.valid_texts = [self.valid_texts[idx] for idx in index]
                self.valid_mask = [self.valid_mask[idx] for idx in index]
        else:
            pass
