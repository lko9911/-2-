import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import cv2
from tqdm import tqdm

class CityscapesDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # 이미지 및 라벨 파일 목록 생성
        self.image_files = sorted(os.listdir(images_dir))[:300]  # 처음 300개 샘플만 사용
        self.label_files = sorted([
            f for f in os.listdir(labels_dir)
            if f.endswith('labelIds.png')
        ])[:300]  # 처음 300개 샘플만 사용

        # 이미지와 라벨 수 확인
        if len(self.image_files) != len(self.label_files):
            raise ValueError("이미지와 라벨의 수가 일치하지 않습니다.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지 로드
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 라벨 로드
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # 변환 적용
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        # numpy에서 torch tensor로 변환
        image = torch.tensor(image).permute(2, 0, 1)  # HWC -> CHW
        label = torch.tensor(label).unsqueeze(0).long()  # HWC -> CHW

        return image, label

    
# U-Net 모델 정의
class UnetGenerator(nn.Module):
    def __init__(self, in_dim=3, out_dim=34, num_filter=64):  # 기본 매개변수 추가
        super(UnetGenerator, self).__init__()
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        # 다운샘플링
        self.down_1 = self.conv_block_2(in_dim, num_filter, act_fn)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_2 = self.conv_block_2(num_filter, num_filter * 2, act_fn)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_3 = self.conv_block_2(num_filter * 2, num_filter * 4, act_fn)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_4 = self.conv_block_2(num_filter * 4, num_filter * 8, act_fn)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 브릿지 레이어
        self.bridge = self.conv_block_2(num_filter * 8, num_filter * 16, act_fn)

        # 업샘플링
        self.trans_1 = self.conv_trans_block(num_filter * 16, num_filter * 8, act_fn)
        self.up_1 = self.conv_block_2(num_filter * 16, num_filter * 8, act_fn)
        self.trans_2 = self.conv_trans_block(num_filter * 8, num_filter * 4, act_fn)
        self.up_2 = self.conv_block_2(num_filter * 8, num_filter * 4, act_fn)
        self.trans_3 = self.conv_trans_block(num_filter * 4, num_filter * 2, act_fn)
        self.up_3 = self.conv_block_2(num_filter * 4, num_filter * 2, act_fn)
        self.trans_4 = self.conv_trans_block(num_filter * 2, num_filter, act_fn)
        self.up_4 = self.conv_block_2(num_filter * 2, num_filter, act_fn)

        # 출력 레이어
        self.out = nn.Sequential(
            nn.Conv2d(num_filter, out_dim, kernel_size=3, stride=1, padding=1),
            nn.LogSoftmax(dim=1),  # LogSoftmax 사용
        )

    def conv_block(self, in_dim, out_dim, act_fn):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            act_fn,
        )

    def conv_trans_block(self, in_dim, out_dim, act_fn):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_dim),
            act_fn,
        )

    def conv_block_2(self, in_dim, out_dim, act_fn):
        return nn.Sequential(
            self.conv_block(in_dim, out_dim, act_fn),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, input):
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_4], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_3], dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_2], dim=1)
        up_3 = self.up_3(concat_3)
        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4, down_1], dim=1)
        up_4 = self.up_4(concat_4)

        out = self.out(up_4)
        return out

# 모델 로드
model_path = 'unet_cityscapes.pth'
model = UnetGenerator(in_dim=3, out_dim=34, num_filter=64)  # 적절한 인자로 초기화
model.load_state_dict(torch.load(model_path))
model.eval()  # 평가 모드로 전환

# 테스트셋 경로 설정
test_images_dir = 'dataset_city1/test/images'
test_labels_dir = 'dataset_city1/test/labels'  # 라벨 디렉토리 경로
test_dataset = CityscapesDataset(test_images_dir, test_labels_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 평가 함수 정의
def evaluate(model, data_loader):
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # 클래스 예측
            all_predictions.append(predicted.numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_predictions), np.concatenate(all_labels)

# 평가 수행
predictions, labels = evaluate(model, test_loader)

# 결과를 이미지로 저장 (선택 사항)
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

for i in range(predictions.shape[0]):
    output_image = predictions[i].astype(np.uint8)  # 예측 결과를 uint8로 변환
    cv2.imwrite(os.path.join(output_dir, f'predicted_{i}.png'), output_image)

print("Evaluation completed!")