import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# U-Net 블록 구성
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
    return nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

def conv_block_2(in_dim, out_dim, act_fn):
    return nn.Sequential(
        conv_block(in_dim, out_dim, act_fn),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )

# U-Net 모델 정의
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
            nn.LogSoftmax(dim=1),
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

# 정규화된 이미지를 원래 값으로 변환하는 함수
def denormalize(image):
    image = (image - image.min()) / (image.max() - image.min())
    return image

# 원본 이미지, Ground Truth 마스크, 예측 마스크를 시각화하여 오버레이하는 함수
def overlay_masks(original, mask, prediction, num_images=1):  # num_images 기본값 수정
    plt.figure(figsize=(15, num_images * 5))
    
    for i in range(num_images):
        # 원본 이미지 역정규화 (배치 차원 제거)
        original_image = denormalize(original[0].permute(1, 2, 0).cpu().numpy())

        # 마스크와 예측 마스크 준비
        ground_truth = mask[0].cpu().numpy()  
        predicted_mask = torch.argmax(prediction[0], dim=0).cpu().numpy()  

        # 고유 값 확인
        print("Ground Truth Unique Values:", np.unique(ground_truth))
        print("Predicted Mask Unique Values:", np.unique(predicted_mask))

        # 예측 마스크 오버레이
        overlay_pred = original_image.copy()
        overlay_pred[predicted_mask == 1] = [1, 0, 0]  # 빨간색으로 오버레이

        # Ground Truth 마스크 오버레이
        overlay_truth = original_image.copy()
        overlay_truth[ground_truth == 90] = [1, 0, 0]  # 녹색으로 오버레이

        # 이미지 시각화
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(overlay_truth)
        plt.title("Ground Truth Mask Overlay")
        plt.axis("off")

        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(overlay_pred)
        plt.title("Predicted Mask Overlay")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# 이미지 로드 및 전처리 (배치 생성)
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (2048, 1024))
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return image

# 마스크 파일을 불러올 때 적용할 전처리 함수
def preprocess_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (2048, 1024))
    mask = torch.from_numpy(mask).long().unsqueeze(0)
    return mask

# 모델 예측 수행 및 오버레이 시각화 호출
def predict_and_overlay(model, image_path, mask_path):
    image = preprocess_image(image_path)
    mask = preprocess_mask(mask_path)
    
    with torch.no_grad():
        output = model(image)
        
        # 시각화 함수에 이미지, 실제 마스크, 모델 예측 결과 전달
        overlay_masks(image, mask, output)

# 모델 경로 및 테스트 이미지 설정
model_path = 'best_unet_cityscapes_ss.pth'
model = UnetGenerator(in_dim=3, out_dim=2, num_filter=64)
model.load_state_dict(torch.load(model_path))
model.eval()

# 테스트할 이미지와 실제 마스크 설정
image_path = 'content/val/images/val_image (1).png'
mask_path = 'content/val/labels/val_labels (1).png'  # 레이블 파일 사용

predict_and_overlay(model, image_path, mask_path)
