import os, sys
import time
import torch 
import utils
import argparse
import importlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from MIML import MIML, Net
from preprocess import AudioDataset
torch.cuda.set_device(1)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def parse_args(verbose=True, logger=None):
    parser = argparse.ArgumentParser(description="Train MIML.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of bacth size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--optimizer", nargs='?', default="adam", help="Specify an optimizer, adagrad, adam, rmsprop, sgd")
    parser.add_argument("--init_type", nargs='?', default="normal", help="normal, xavier, kaiming")
    parser.add_argument("--train_data_path", nargs='?', default="/data1/xyf/train_data.npy", help="normal, xavier, kaiming")
    parser.add_argument("--labels_path", nargs='?', default="/data1/xyf/label.npy", help="normal, xavier, kaiming")
    parser.add_argument("--M", type=int, default=25, help="The number of bases.")
    parser.add_argument("--F", type=int, default=1537, help="The dimensions of basis vector after NMF.")
    parser.add_argument("--fc_dimensions", type=int, default=512, help="The dimensions of basis vector after reduction.")
    parser.add_argument("--K", type=int, default=3, help="The number of sub-concepts.")
    parser.add_argument("--L", type=int, default=8, help="The number of object categories.")
    parser.add_argument("--gradient_clip", type=float, default=5.0, help="Gradient clip value.")
    args = parser.parse_args()
    if verbose:
        logger = logger #if logger is not None else print
        logger('epochs              | %d' % args.epochs)
        logger('batch_size          | %d' % args.batch_size)
        logger('lr                  | %f' % args.lr)
        logger('optimizer           | %s' % args.optimizer)
        logger('M                   | %s' % args.M)
        logger('F                   | %s' % args.F)
        logger('fc_dimensions       | %s' % args.fc_dimensions)
        logger('K                   | %s' % args.K)
        logger('L                   | %s' % args.L)
        logger('gradient clip       | %s' % args.gradient_clip)
    return args

def get_optim(model, args):
    if args.optimizer.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise NotImplementedError("%s optimizer not supprted!" % args.optimizer)
    return optimizer

def evaluate(model, dataset):
    with torch.no_grad():
        model.eval()
        X_test = torch.from_numpy(dataset.X_train)
        X_test = X_test.type(torch.FloatTensor).to(device)
        predict,_ = model(X_test)
        #print(predict[0])
        predict = predict.cpu().numpy()
        label = dataset.y_train
        predict = np.argmax(predict, axis=1)
        label = np.argmax(label, axis=1)
        model.train()
        #print(predict)
        #print(label)
        #print(predict.shape)
        #print(label.shape)
        return accuracy_score(predict, label)
    
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.06 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    OUTPUT_DIR = time.strftime('%Y%m%d%H%M', time.localtime())
    if os.path.exists(OUTPUT_DIR):
        os.system("rm -rf %s" % OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)
    os.mkdir(os.path.join(OUTPUT_DIR, 'models'))
    os.system("cp run.sh %s" % OUTPUT_DIR)
    log_fp = open(os.path.join(OUTPUT_DIR, 'log.txt'), 'w')
    logger = utils.Logger(log_fp)
    logger("Init models.")
    args = parse_args(verbose=True, logger=logger)
    model = MIML(args).to(device)
    #model = Net().to(device)
    optimizer = get_optim(model, args)
    logger("Load dataset.")
    dataset = AudioDataset(args.train_data_path, args.labels_path, args.batch_size)
    logger("Begin Training.")
    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.NLLLoss()
    accumulate_loss = []
    best_accuracy = 0
    best_epoch = -1
    for epoch in range(args.epochs):
        # train
        model.train()
        adjust_learning_rate(optimizer, epoch)
        # dataset.shuffle(subset="train")
        for i in range(dataset.train_batch_count):
            data = dataset.get_batch(i, subset="train")
            x_train = torch.from_numpy(data["X_train"])
            x_train = x_train.type(torch.FloatTensor).to(device)
            y_train = torch.from_numpy(data["y_train"])
            #y_train = y_train.type(torch.LongTensor).to(device)
            y_train = torch.argmax(y_train, dim=-1).to(device)
            #print(y_train.shape)
            output,_ = model(x_train)
            # print(output.shape)
            # print(y_train.shape)
            loss = loss_function(output, y_train)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            accumulate_loss.append(loss.item())
            if i % 5 == 5-1:
                mean_loss = sum(accumulate_loss) / len(accumulate_loss)
                logger("Epoch %d, step %d / %d, loss = %f" % (epoch+1, i+1, dataset.train_batch_count, mean_loss))
                accumulate_loss = []
        accuracy = evaluate(model, dataset)
        logger("Epoch %d, test accuracy = %.4f" % (epoch+1, accuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            #torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'snapshot.model'))
            torch.save(model, os.path.join(OUTPUT_DIR, 'snapshot.model'))
            logger("Epoch %d, save model." % (epoch + 1))
            
    
