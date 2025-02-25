import numpy as np
import cv2
import os
from skimage.transform import radon, iradon

class SIRTReconstruction:
    def __init__(self, sinograms, num_projections=180, iterations=10, learning_rate=0.1):
        """
        Simultaneous Iterative Reconstruction Technique (SIRT) 알고리즘
        :param sinograms: Forward Projection에서 생성된 Sinogram 리스트
        :param num_projections: Projection 개수
        :param iterations: Iterative 업데이트 횟수
        :param learning_rate: 업데이트 학습률 (λ)
        """
        self.sinograms = sinograms
        self.num_projections = num_projections
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.reconstructed_images = []

    def reconstruct(self):
        """SIRT 알고리즘을 적용하여 Reconstruction 수행"""
        theta = np.linspace(0., 180., self.num_projections, endpoint=False)  # Projection 각도 설정

        for sinogram in self.sinograms:
            # 초기 Reconstruction 이미지를 0으로 설정
            recon_img = np.zeros((sinogram.shape[0], sinogram.shape[0]))

            for _ in range(self.iterations):  # Iteration 진행
                # 전체 Projection을 한 번에 고려하여 업데이트
                Ai = radon(recon_img, theta=theta, circle=True)  # 현재 상태에서 Forward Projection 수행
                correction = sinogram - Ai  # Projection 차이 계산
                backprojected = iradon(correction, theta=theta, filter_name=None, circle=True)  # Back Projection 수행
                
                # Reconstruction 업데이트
                recon_img += self.learning_rate * backprojected
            
            self.reconstructed_images.append(recon_img)

        self.reconstructed_images = np.array(self.reconstructed_images)
        return self.reconstructed_images

    def save_reconstructed_images(self, save_dir):
        """복원된 이미지를 저장"""
        os.makedirs(save_dir, exist_ok=True)
        for idx, img in enumerate(self.reconstructed_images):
            save_path = os.path.join(save_dir, f"sirt_reconstructed_{idx}.png")
            cv2.imwrite(save_path, (img / np.max(img) * 255).astype(np.uint8))

# 실행 예제
if __name__ == "__main__":
    from forward_projection import ForwardProjection
    from data_loader import DataLoader

    # 데이터 로드 및 Mass 분할
    loader = DataLoader("data/sample_ct.png", mass_size=64, overlap=16)
    loader.load_image()
    masses = loader.split_into_masses()

    # Forward Projection 실행 (Sinogram 생성)
    projector = ForwardProjection(masses, num_projections=180)
    sinograms = projector.compute_sinogram()

    # SIRT Reconstruction 수행
    print("🔹 SIRT Iterative Reconstruction 실행 중...")
    sirt_recon = SIRTReconstruction(sinograms, num_projections=180, iterations=10, learning_rate=0.1)
    reconstructed_images = sirt_recon.reconstruct()

    # 결과 저장
    output_dir = "results/sirt_reconstructed"
    sirt_recon.save_reconstructed_images(output_dir)
    print(f"✅ SIRT Reconstruction 완료! ({len(reconstructed_images)}개 이미지 저장됨)")
