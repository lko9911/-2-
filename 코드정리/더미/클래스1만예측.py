import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def conv_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def conv_trans_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

def conv_block_2(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model

class UnetGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, num_filter):
        super(UnetGenerator, self).__init__()
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        self.down_1 = conv_block_2(in_dim, num_filter, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = conv_block_2(num_filter, num_filter * 2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = conv_block_2(num_filter * 2, num_filter * 4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = conv_block_2(num_filter * 4, num_filter * 8, act_fn)
        self.pool_4 = maxpool()

        self.bridge = conv_block_2(num_filter * 8, num_filter * 16, act_fn)

        self.trans_1 = conv_trans_block(num_filter * 16, num_filter * 8, act_fn)
        self.up_1 = conv_block_2(num_filter * 16, num_filter * 8, act_fn)
        self.trans_2 = conv_trans_block(num_filter * 8, num_filter * 4, act_fn)
        self.up_2 = conv_block_2(num_filter * 8, num_filter * 4, act_fn)
        self.trans_3 = conv_trans_block(num_filter * 4, num_filter * 2, act_fn)
        self.up_3 = conv_block_2(num_filter * 4, num_filter * 2, act_fn)
        self.trans_4 = conv_trans_block(num_filter * 2, num_filter, act_fn)
        self.up_4 = conv_block_2(num_filter * 2, num_filter, act_fn)

        self.out = nn.Sequential(
            nn.Conv2d(num_filter, out_dim, kernel_size=3, stride=1, padding=1),
            nn.LogSoftmax(dim=1),  # LogSoftmax 사용
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

# 클래스 수 설정 및 색상 팔레트 정의 (클래스마다 임의의 색상 부여)
num_classes = 34
colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

def apply_color_map(segmentation, colors):
    color_segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    for label in range(num_classes):
        color_segmentation[segmentation == label] = colors[label]
    return color_segmentation

# 이미지 로드 및 전처리
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (2048, 1024))  # 모델에 맞게 크기 조정
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # 배치 추가 및 정규화
    return image

# 예측 수행 및 시각화
def predict_and_visualize(model, image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted = predicted.squeeze().cpu().numpy()
        
        # 색상 맵 적용
        color_mapped_segmentation = apply_color_map(predicted, colors)

        # 검출된 클래스 카운트 출력
        unique_classes, counts = np.unique(predicted, return_counts=True)
        print("Detected classes and their counts:")
        for cls, count in zip(unique_classes, counts):
            print(f"Class {cls}: {count} pixels")
        
        # 원본 이미지와 세그멘테이션 결과 나란히 표시
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(cv2.imread(image_path)[:, :, ::-1])
        ax[0].set_title("Original Image")
        ax[1].imshow(color_mapped_segmentation)
        ax[1].set_title("Segmented Image")
        plt.show()

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# 나머지 코드 동일

# 클래스 1 픽셀만 남기고 다른 클래스는 배경으로 설정하는 함수
def filter_class(segmentation, target_class=1):
    filtered = np.zeros_like(segmentation)
    filtered[segmentation == target_class] = target_class
    return filtered

# 예측 수행 및 시각화
def predict_and_visualize_single_class(model, image_path, target_class=7):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted = predicted.squeeze().cpu().numpy()
        
        # 클래스 1만 검출
        filtered_segmentation = filter_class(predicted, target_class=target_class)
        color_mapped_segmentation = apply_color_map(filtered_segmentation, colors)

        # 클래스 1의 픽셀 수 출력
        class_1_count = np.sum(filtered_segmentation == target_class)
        print(f"Class {target_class}: {class_1_count} pixels")
        
        # 원본 이미지와 세그멘테이션 결과 나란히 표시
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(cv2.imread(image_path)[:, :, ::-1])
        ax[0].set_title("Original Image")
        ax[1].imshow(color_mapped_segmentation)
        ax[1].set_title(f"Segmented Class {target_class} Only")
        plt.show()

# 모델 및 테스트 이미지 경로 설정
model_path = 'unet_cityscapes.pth'
model = UnetGenerator(in_dim=3, out_dim=num_classes, num_filter=64)
model.load_state_dict(torch.load(model_path))
model.eval()

# 이미지 경로 (사용할 이미지 경로로 변경)
image_path = 'dataset_city1/train/images/aachen_000173_000019_leftImg8bit.png'

# 클래스 1만 검출
predict_and_visualize_single_class(model, image_path, target_class=7)

