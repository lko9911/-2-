# 라이브러리 가져오기 _ 전이학습용 (sklearn, time, torch)
import os
from sklearn.model_selection import train_test_split
from PIL import Image # 전치리 전용 (커스텀)
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import copy

# 학습 완료 성능 평가 자료
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터셋 구성 클래스 (단일 분류) : 검증셋, 테스트셋용
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None): # 인스턴스 자동 초기화 (불러올 때)
        self.image_paths = image_paths # 이미지 경로 정보
        self.labels = labels # 라벨링 정보
        self.transform = transform # 변환 초기화

    def __len__(self):
        return len(self.image_paths) # 이미지 자료의 길이 (배수 적용 코드)

    def __getitem__(self, idx): # 전처리 코드
        image = Image.open(self.image_paths[idx]).convert('RGB') # RGB값으로 전처리
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 데이터셋 구성 클래스 (단일 분류) : 학습셋용 (2배수 적용)
class CustomDataset_train(Dataset):
    def __init__(self, image_paths, labels, transform=None): # 인스턴스 자동 초기화 (불러올 때)
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths) * 2 # 배수 적용

    def __getitem__(self, idx): # 인덱스 요소 재설정
        actual_idx = idx % len(self.image_paths)
        image = Image.open(self.image_paths[actual_idx]).convert('RGB')
        label = self.labels[actual_idx] * 2
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 데이터셋 로드 (jpg, jpeg, png) : 데이터 셋은 data_dir이름을 가지고 폴더명이 클래스 명과 동일할 떄 사용
data_dir = 'dataset'  # 이미지 폴더 경로
class_names = os.listdir(data_dir) # 클래스명 인덱스로 가져오기
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

image_paths = []
labels = []

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if img_path.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(img_path)
                labels.append(class_to_idx[class_name])

# 데이터 증강 _ 파이토치 튜토리얼 변형
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 모델 학습 함수 (디폴트 : 에포크25, 배치사이즈 32)
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

#---------------------------------------------모델 평가---------------------------------------------#
# 혼동행렬 출력 
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_preds = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return all_labels, all_preds

def plot_confusion_matrix(labels, preds, classes):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

#---------------------------------------------메인 함수---------------------------------------------#
def main():
    # 데이터셋 분할 학습 0.8  검증 0.1 테스트 0.1
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42, stratify=labels)
    val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)
    
    # 데이터셋 구성
    train_dataset = CustomDataset_train(train_paths, train_labels, transform=data_transforms['train'])
    val_dataset = CustomDataset(val_paths, val_labels, transform=data_transforms['val'])
    test_dataset = CustomDataset(test_paths, test_labels, transform=data_transforms['test'])

    # 데이터 로더(학습셋만 셔플, num_worker=2 디폴트)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    ## CUDA 사용
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 백본 모델 ResNet50 불러오기
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # 학습 정보 구성
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    # 손실 함수 및 옵티마이저 설정 : 학습률 0.001 이외 튜토리얼 기본 값
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # 모델 학습 (에포크 100 구성)
    model_ft = train_model(model_ft, criterion, optimizer_ft, dataloaders, dataset_sizes, device, num_epochs=101)

    # 모델 저장
    torch.save(model_ft.state_dict(), 'resnet50_species.pth')

    # 테스트 셋에 대한 평가 및 혼동행렬 출력
    test_labels, test_preds = evaluate_model(model_ft, dataloaders['test'], criterion, device)
    plot_confusion_matrix(test_labels, test_preds, class_names)

if __name__ == '__main__':
    main()