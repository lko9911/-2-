import cv2
import numpy as np

# 레이블 이미지 로드 (BGR 형식으로 로드)
label_image = cv2.imread("dataset_city2/train/labels/aachen_000000_000019_gtFine_color.png")

# 추출할 색상 정의
target_color = np.array([128, 64, 128])  # (B, G, R) 형식

# 새로운 마스크 이미지 생성 (모든 픽셀을 검은색으로 초기화)
mask_image = np.zeros_like(label_image, dtype=np.uint8)

# 색상 마스크 생성
mask = np.all(label_image == target_color, axis=-1)  # 지정된 색상과 일치하는 픽셀 마스크 생성

# 마스크에서 지정된 색상 위치를 흰색으로 설정
mask_image[mask] = [255, 255, 255]  # 흰색

# 마스크 이미지 저장
cv2.imwrite("color_mask_image.png", mask_image)
