# 基于MIML和Audio Basis的音源定位和分解
这是清华大学电子系大三《视听信息系统导论》的大作业，由无63的徐远帆和李明哲共同完成。具体原理参考自[《Learning to Separate Object Sounds by Watching Unlabeled Video》](https://arxiv.org/pdf/1804.01665.pdf)。
## 环境依赖
python        2.7  
torch         0.4.1  
torchvison    0.2.1  
audioread     2.1.6  
librosa       0.6.2  
musdb         0.2.3  
museval       0.2.0  
numpy         1.15.4  
opencv-python 3.4.3  
pydub         0.23.0  
scikit-learn  0.20.0  
scipy         1.1.0  
也可直接conda install requirements.txt  

## 文件清单
model/prepare_data.py：处理提供的音频，得到训练数据，处理好的训练数据在train_data.zip里  
model/MIML.py：MIML网络部分，用PyTorch实现。  
model/train.py：训练部分，会用到preprocess.py和utils.py等预处理。  
model/generate_base.py：由训练好的模型提取出Audio Basis，201812132050文件夹里提供了一个训练好的模型。  
bases/*：提取出的Audio Basis。  
decompose.py：音源定位和分解函数。  
Evaluate.py：评测函数，返回SDR和Accuracy。  
feat_extractor.py：用于提取图像特征。  
gt_json_gen.py：生成result_json文件。  
log/*：记录运行结果。  
result_audio/*：保存着经过调整后较为理想、结果较好的一组音频。  
result_audio_test/*：保存助教运行decompose.py生成的音频。  
result_json/*：保存gt.json和result.json。  
testset7_result_audio/*：保存testset7中分离的音频。  
## 运行流程
1. 将dataset、testset25、testset7、gt_audio放在当前目录下。
2. 如果要重头开始训练，cd进入model/，运行prepare_data.py，生成训练数据（需要大量时间，也可从清华云盘下载），然后修改train.py里的train data和label的路径，运行train.py开始训练。模型训练好后会自动保存，然后修改generate_base.py里的模型路径，运行generate_base.py生成不同乐器的Audio basis，即bases/*里的npy格式数据。
3. 如果不想训练模型，可直接使用预训练好的bases/*，运行decompose.py即可产生分解后的音频，保存在result_audio_test/中。
4. 运行Evaluate.py，得到评测后的SDR和Accuracy。


