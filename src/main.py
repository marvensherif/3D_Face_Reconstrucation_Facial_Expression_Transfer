import argparse
import os
import cv2
from helper import detect_face_landmarks, transfer_expression, save_2d_landmarks, save_3d_landmarks_html, compute_rmse, compute_ssim, compute_mae_3d
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_image', type=str, help="Path to the source image")
    parser.add_argument('target_image', type=str, help="Path to the target image")
    parser.add_argument('--output_dir', type=str, default='output/', help="Directory to save processed images (default: output/)")
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    source_img = cv2.imread(args.source_image)
    target_img = cv2.imread(args.target_image)

    if source_img is None or target_img is None:
        print("Error: One or both images could not be loaded.")
        return

    if source_img.shape != target_img.shape:
        target_img = cv2.resize(target_img, (source_img.shape[1], source_img.shape[0]))


    source_landmarks_2d, source_landmarks_3d = detect_face_landmarks(source_img)
    target_landmarks_2d, target_landmarks_3d = detect_face_landmarks(target_img)

    if source_landmarks_2d is None or target_landmarks_2d is None:
        print("Error: Could not detect landmarks in one or both images.")
        return

    # Save 2D landmarks
    save_2d_landmarks(source_img.copy(), source_landmarks_2d, os.path.join(args.output_dir, "source_2d_landmarks.jpg"))
    save_2d_landmarks(target_img.copy(), target_landmarks_2d, os.path.join(args.output_dir, "target_2d_landmarks.jpg"))

    # Save 3D landmarks as HTML
    save_3d_landmarks_html(source_landmarks_3d, os.path.join(args.output_dir, "source_3d_landmarks.html"))
    save_3d_landmarks_html(target_landmarks_3d, os.path.join(args.output_dir, "target_3d_landmarks.html"))

    # Perform facial expression transfer
    target_img_with_expression = transfer_expression(source_img, target_img.copy(), source_landmarks_2d, target_landmarks_2d)

    # Save the final image after expression transfer
    cv2.imwrite(os.path.join(args.output_dir, "target_image_after_expression_transfer.jpg"), target_img_with_expression)

    # Calculate quality metrics
    rmse_2d = compute_rmse(source_landmarks_2d, target_landmarks_2d)
    mae_3d = compute_mae_3d(source_landmarks_3d, target_landmarks_3d)
    ssim_value = compute_ssim(source_img, target_img_with_expression)

    # Print metrics
    print(f"RMSE (2D Landmarks): {rmse_2d}")
    print(f"MAE (3D Landmarks): {mae_3d}")
    print(f"SSIM (Structural Similarity Index): {ssim_value}")

if __name__ == "__main__":
    main()
