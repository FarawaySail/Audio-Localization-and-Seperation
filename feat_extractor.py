import os
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json

global net
global normalize
global preprocess
global features_blobs
global classes
global weight_softmax
labels_path='labels.json'
idxs=[401,402,486,513,558,642,776,889]
names=['accordion','acoustic_guitar','cello','trumpet','flute','xylophone','saxophone','violin']

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def hook_feature(module, input, output):
    global features_blobs
    features_blobs=output.data.cpu().numpy()

def load_model():
    global net
    global normalize
    global preprocess
    global features_blobs
    global classes
    global weight_softmax
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'
    net.eval()
    net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
       transforms.Resize((224,448)),
       transforms.ToTensor(),
       normalize
    ])
    classes = {int(key):value for (key, value)
              in json.load(open(labels_path,'r')).items()}
    if torch.cuda.is_available():
        net=net.cuda()

def get_CAM(imdir,savedir,imname):
    img_pil = Image.open(os.path.join(imdir,imname))
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    if torch.cuda.is_available():
        img_variable=img_variable.cuda()
    img = cv2.imread(os.path.join(imdir,imname))
    height, width, _ = img.shape
    logit_left = net(img_variable[:,:,:,:224])
    logit_right = net(img_variable[:,:,:,224:])
    h_left = F.softmax(logit_left, dim=1).data.squeeze()
    h_right = F.softmax(logit_right, dim=1).data.squeeze()
    if torch.cuda.is_available():
        h_left=h_left.cpu()
        h_right=h_right.cpu()
    probs1 = h_left.numpy()
    probs2 = h_right.numpy()
    probs_left=[]
    probs_right=[]
    for i in range(0, 8):
        # print('{:.3f} -> {}'.format(probs1[idxs[i]], names[i]))
        '''
        CAMs = returnCAM(features_blobs, weight_softmax, [idxs[i]])        
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(os.path.join(savedir,names[i],imname), result)
        '''
        probs_left.append(probs1[idxs[i]])
        probs_right.append(probs2[idxs[i]])
    return probs_left, probs_right

def main():
    imdir='/home/eric/std/dataset/images/duet/acoustic_guitarviolin/2'
    load_model()
    imlist=os.listdir(imdir)
    probs=np.zeros([8])
    for im in imlist:
        probs1=get_CAM(imdir,'results',im)
        probs=probs+np.array(probs1)
    print probs
    print names

if __name__=='__main__':
    main()
            

