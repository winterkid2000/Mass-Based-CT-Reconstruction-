import numpy as np
import cv2
import os

class OverlappingBlender:
    def __init__(self, reconstructed_masses, mass_size=64, overlap=16, image_size=512):
        """
        Overlapping된 Mass 블록을 Blending하여 최종 Reconstruction 이미지 생성
        :param reconstructed_masses: Mass 단위로 Reconstruction된 이미지들 (List)
        :param mass_size: Mass 크기 (M×M)
        :param overlap: Overlapping 크기 (O)
        :param image_size: 최종 Reconstruction 이미지 크기 (N×N)
        """
        self.reconstructed_masses = reconstructed_masses
        self.mass_size = mass_size
        self.overlap = overlap
        self.image_size = image_size
        self.final_image = np.zeros((image_size, image_size), dtype=np.float32)
        self.weight_matrix = np.zeros((image_size, image_size), dtype=np.float32)

    def apply_blending(self):
        """Overlapping된 Mass 블록을 Blending하여 최종 이미지 생성"""
        idx = 0
        step_size = self.mass_size - self.overlap  # Overlapping 고려한 이동 크기

        for i in range(0, self.image_size - self.mass_size + 1, step_size):
            for j in range(0, self.image_size - self.mass_size + 1, step_size):
                # Gaussian Blending 적용
                mass = self.reconstructed_masses[idx]
                weight_mask = self._generate_gaussian_mask(self.mass_size)

                self.final_image[i:i+self.mass_size, j:j+self.mass_size] += mass * weight_mask
                self.weight_matrix[i:i+self.mass_size, j:j+self.mass_size] += weight_mask
                idx += 1

        # 가중치 정규화
        self.final_image /= (self.weight_matrix + 1e-8)  # 0으로 나누는 것 방지
        return self.final_image

    def _generate_gaussian_mask(self, size):
        """Gaussian Blending을 위한 가중치 마스크 생성"""
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, y)
        gaussian_mask = np.exp(-(xx**2 + yy**2))  # 2D Gaussian 적용
        return gaussian_mask / np.max(gaussian_mask)  # 정규화

    def save_final_image(self, save_path):
        """Blended된 최종 Reconstruction 이미지를 저장"""
        cv2.imwrite(save_path, (self.final_image / np.max(self.final_image) * 255).astype(np.uint8))

# 실행 예제
if __name__ == "__main__":
    from iterative_reconstruction import ARTReconstruction
    from forward_projection import ForwardProjection
    from data_loader import DataLoader

    # 데이터 로드 및 Mass 분할
    loader = DataLoader("data/sample_ct.png", mass_size=64, overlap=16)
    loader.load_image()
    masses = loader.split_into_masses()

    # Forward Projection 실행 (Sinogram 생성)
    projector = ForwardProjection(masses, num_projections=180)
    sinograms = projector.compute_sinogram()

    # Iterative Reconstruction (ART) 수행
    art_recon = ARTReconstruction(sinograms, num_projections=180, iterations=10, learning_rate=0.1)
    reconstructed_masses = art_recon.reconstruct()

    # Overlapping Blending 수행
    print("🔹 Overlapping Blending 실행 중...")
    blender = OverlappingBlender(reconstructed_masses, mass_size=64, overlap=16, image_size=512)
    final_image = blender.apply_blending()

    # 결과 저장
    output_path = "results/final_reconstructed.png"
    blender.save_final_image(output_path)
    print(f"✅ 최종 Reconstruction 완료! ({output_path}에 저장됨)")
