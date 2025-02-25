import numpy as np
import cv2
import os
import pydicom
from skimage.transform import resize

class DataLoader:
    def __init__(self, image_path, mass_size=64, overlap=16, normalize=True):
        self.image_path = image_path
        self.mass_size = mass_size  # M 크기
        self.overlap = overlap  # O 크기
        self.normalize = normalize
        self.image = None  # 로드된 원본 이미지
        self.masses = []  # Mass 단위로 나눈 블록 저장

    def load_image(self):
        """CT 이미지를 로드하고, N×N 크기로 변환"""
        if self.image_path.lower().endswith('.dcm'):  # DICOM 파일인 경우
            dicom_data = pydicom.dcmread(self.image_path)
            image = dicom_data.pixel_array
        else:  # PNG, JPG 같은 일반 이미지인 경우
            image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        # 정규화 (0~1 범위)
        if self.normalize:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))

        self.image = image
        return self.image

    def split_into_masses(self):
        """이미지를 M×M 크기의 Mass 단위로 분할 (Overlapping 포함)"""
        if self.image is None:
            raise ValueError("이미지를 먼저 로드해야 합니다!")

        N = self.image.shape[0]  # 원본 이미지 크기 (512×512 또는 1024×1024 등)
        M = self.mass_size
        O = self.overlap  # Overlapping 크기

        masses = []
        for i in range(0, N - M + 1, M - O):  # Overlapping 고려하여 슬라이딩
            for j in range(0, N - M + 1, M - O):
                mass = self.image[i:i+M, j:j+M]
                masses.append(mass)

        self.masses = np.array(masses)
        return self.masses

    def save_masses(self, save_dir):
        """Mass 블록을 이미지 파일로 저장 (디버깅용)"""
        os.makedirs(save_dir, exist_ok=True)
        for idx, mass in enumerate(self.masses):
            save_path = os.path.join(save_dir, f"mass_{idx}.png")
            cv2.imwrite(save_path, (mass * 255).astype(np.uint8))

# 실행 예제
if __name__ == "__main__":
    loader = DataLoader("data/sample_ct.png", mass_size=64, overlap=16)
    image = loader.load_image()
    masses = loader.split_into_masses()
    loader.save_masses("results/masses")  # 분할된 Mass 저장
