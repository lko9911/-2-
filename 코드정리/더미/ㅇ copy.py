import cv2
import numpy as np

# 레이블 이미지 로드 (BGR 형식으로 로드)
label_image = cv2.imread("dataset_city2/train/labels/aachen_000000_000019_gtFine_color.png")

# 색상 맵 정의
color_map = {
    0: [0, 0, 255],          # 배경: 검은색
    1: [111, 74, 0],       # static
    2: [81, 0, 81],        # ground
    3: [128, 64, 127],     # road
    4: [244, 35, 232],     # sidewalk
    5: [250, 170, 160],    # parking
    6: [230, 150, 140],    # rail track
    7: [70, 70, 70],       # building
    8: [102, 102, 156],    # wall
    9: [190, 153, 153],    # fence
    10: [180, 165, 180],    # guard rail
    11: [150, 100, 100],    # bridge
    12: [150, 120, 90],     # tunnel
    13: [153, 153, 153],    # pole
    14: [153, 153, 153],    # polegroup
    15: [250, 170, 30],     # traffic light
    16: [220, 220, 0],       # traffic sign
    17: [107, 142, 35],      # vegetation
    18: [152, 251, 152],     # terrain
    19: [70, 130, 180],      # sky
    20: [220, 20, 60],       # person
    21: [255, 0, 0],         # rider
    22: [0, 0, 142],         # car
    23: [0, 0, 70],          # truck
    24: [0, 60, 100],        # bus
    25: [0, 0, 90],          # caravan
    26: [0, 0, 110],         # trailer
    27: [0, 80, 100],        # train
    28: [0, 0, 230],         # motorcycle
    29: [119, 11, 32],       # bicycle
    30: [0, 0, 142],         # license plate
    31: [255, 255, 0],       # yellow (추가)
    32: [0, 255, 255],       # cyan (추가)
    33: [255, 0, 255],       # magenta (추가)
    34: [192, 192, 192]      # silver (추가)
}

# 새로운 이미지를 생성
height, width, _ = label_image.shape
color_label_image = np.zeros((height, width, 3), dtype=np.uint8)

# 각 픽셀에 대해 색상 할당
for class_id, color in color_map.items():
    # 색상 변환 (BGR을 RGB로 바꿔서 비교)
    mask = np.all(label_image == color_map[class_id], axis=-1)  # 현재 클래스에 대한 마스크 생성
    color_label_image[mask] = color  # 색상 할당

# 새로운 레이블 이미지 저장
cv2.imwrite("color_label_image.png", color_label_image)
