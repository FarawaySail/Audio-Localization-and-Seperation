import os, sys
import pdb
import time
import torch 
import utils
import argparse
import importlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_base(model, instrument, num):
    data = np.load("/data1/xyf/base/" + instrument + "_train_data.npy")
    idx = np.random.permutation(data.shape[0])
    base = []
    for i in range(num):
        index = idx[i]
        X_test = torch.from_numpy(data[index]).unsqueeze(0)
        X_test = X_test.type(torch.FloatTensor).to(device)
        with torch.no_grad():
            model.eval()
            L, map = model(X_test)
        L = L.view(-1).cpu().numpy()
        Relation_Map = map.squeeze(0).squeeze(0).cpu().numpy()
        index = L.argmax()
        index = Relation_Map[:,index].argmax()
        X_test = X_test.squeeze(0).cpu().numpy()
        W = X_test[:, index]
        base.append(W)
    np.save("./base_aug/"+instrument + "_base.npy", np.array(base),)
    print(instrument + " complete.")
    
    
    
if __name__ == "__main__":
    model_path = "201812132050/snapshot.model"
    model = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0))
    num = 300
    save_base(model, "accordion", num)
    save_base(model, "acoustic_guitar", num)
    save_base(model, "cello", num)
    save_base(model, "flute", num)
    save_base(model, "saxophone", num)
    save_base(model, "trumpet", num)
    save_base(model, "violin", num)
    save_base(model, "xylophone", num)
