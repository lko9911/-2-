#--------------------------------------------라이브러리--------------------------------------------#

## 참고 사이트 https://ingu627.github.io/code/ResNet50_pytorch/
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import transforms
import time
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

# 사전 훈련된 네트워크 불러오기
## self는 클래스의 객체(자기 참조) 객체내 위치 상관없이 참조하기 위해 사용
class MnistResNet(nn.Module):
    def __init__(self, in_channels=1): ## __init()__는 클래스 초기화
        super(MnistResNet, self).__init__() ## 초기화 메서드 호출 함수 : super(), 클래스 호출시 무조건 초기화
        
        # 토치비전에서 모델 가져오기, 사전훈련 
        self.model = models.resnet50(pretrained=True)

        # RGB 3채널을 fashion_mnist에 맞게 1로 바꿈(1차원)
        ## 첫번째 신경망 조정 
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 1000개 클래스 대신 10개 클래스로 바꾸기
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs,10)
    
    def forward(self,x): # 모델에 있는 forward 함수 그대로 사용
        return self.model(x)

my_resnet = MnistResNet()

# 사전 정의 네트워크 확인하기
input = torch.randn((16,1,224,224)) # 표준 정규분포로 하여 4차원 텐서 생성
output = my_resnet(input)
print(output.shape)

print(my_resnet)

# cuda 사용
if torch.cuda.is_available():
    device = torch.device("CUDA")
else:
    device = torch.device("CPU")

# Dataloaders 함수 (데이터 전처리 함수 정의)
def Dataloaders(train_batch_size, val_batch_size):
    fashion_mnist = torchvision.datasets.FashionMINIST(
        download=True,
        train=True,
        root=".").train_data.float()
    
    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(), # 이미지 텐서화 (PIL image)
        transforms.Normalize((fashion_mnist.mean()/255,), (fashion_mnist.std()/255,))])
    
    train_loader = DataLoader(torchvision.datasets.FashionMNIST(
        download=True,
        root=".",
        transform=data_transform,
        train=True),
        batch_size = train_batch_size,
        shuffle = True)

    val_loader = DataLoader(torchvision.datasets.FashionMNIST(
        download=True,
        root=".",
        transform=data_transform,
        train=False),
        batch_size = val_batch_size,
        shuffle = False)    
    
    return train_loader, val_loader

# 모델 평가 코드
def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).kwonlyargs:
        # getfullargspec(func) : 호출 가능한 개체의 매개 변수의 이름과 기본값을 가져옴 (튜플로 반환)
        # kwonlyargs : 모든 parameter 값 확인
        return metric_fn(true_y, pred_y, average="macro")
    
    else:
        return metric_fn(true_y, pred_y)
    
def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision","recall","F1","accuracy"),(p,r,f1,a)):
        print(f"\{name.rjust(14,' ')}: {sum(scores)/batch_size:.4f}")

    