import numpy as np
import cv2
from skimage.transform import radon
import os

class ForwardProjection:
    def __init__(self, masses, num_projections=180):
        """
        Mass 단위로 Forward Projection을 수행하여 Sinogram을 생성
        :param masses: Mass 단위로 나눈 작은 블록들 (List 또는 NumPy 배열)
        :param num_projections: Projection 개수 (기본값 180)
        """
        self.masses = masses
        self.num_projections = num_projections
        self.sinograms = []

    def compute_sinogram(self):
        """각 Mass에서 Radon Transform을 적용하여 Sinogram 생성"""
        theta = np.linspace(0., 180., self.num_projections, endpoint=False)  # Projection 각도 설정

        for mass in self.masses:
            sinogram = radon(mass, theta=theta, circle=True)  # Radon Transform 수행
            self.sinograms.append(sinogram)

        self.sinograms = np.array(self.sinograms)  # 리스트 → NumPy 배열 변환
        return self.sinograms

    def save_sinograms(self, save_dir):
        """생성된 Sinogram을 PNG 이미지로 저장"""
        os.makedirs(save_dir, exist_ok=True)
        for idx, sinogram in enumerate(self.sinograms):
            save_path = os.path.join(save_dir, f"sinogram_{idx}.png")
            cv2.imwrite(save_path, (sinogram / np.max(sinogram) * 255).astype(np.uint8))

# 실행 예제
if __name__ == "__main__":
    from data_loader import DataLoader

    # 데이터 로드 및 Mass 분할
    loader = DataLoader("data/sample_ct.png", mass_size=64, overlap=16)
    loader.load_image()
    masses = loader.split_into_masses()

    # Forward Projection 실행
    projector = ForwardProjection(masses, num_projections=180)
    sinograms = projector.compute_sinogram()
    projector.save_sinograms("results/sinograms")  # Sinogram 결과 저장
