import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.8)

def detect_face_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape
        landmarks_array = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark])
        landmarks_3d = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks.landmark])
        return landmarks_array, landmarks_3d
    return None, None

def apply_delaunay(image, landmarks):
    tri = Delaunay(landmarks)
    return tri

def warp_triangle(src_img, dst_img, src_tri, dst_tri):
    src_rect = cv2.boundingRect(np.float32([src_tri]))
    dst_rect = cv2.boundingRect(np.float32([dst_tri]))
    
    src_tri_offset = [(src_tri[i][0] - src_rect[0], src_tri[i][1] - src_rect[1]) for i in range(3)]
    dst_tri_offset = [(dst_tri[i][0] - dst_rect[0], dst_tri[i][1] - dst_rect[1]) for i in range(3)]
    
    src_cropped = src_img[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]
    
    src_mask = np.zeros((src_rect[3], src_rect[2]), dtype=np.uint8)
    dst_mask = np.zeros((dst_rect[3], dst_rect[2]), dtype=np.uint8)
    
    cv2.fillConvexPoly(src_mask, np.int32(src_tri_offset), 255)
    cv2.fillConvexPoly(dst_mask, np.int32(dst_tri_offset), 255)
    
    src_cropped_warped = cv2.warpAffine(src_cropped, 
                                        cv2.getAffineTransform(np.float32(src_tri_offset), np.float32(dst_tri_offset)), 
                                        (dst_rect[2], dst_rect[3]), 
                                        None, 
                                        flags=cv2.INTER_CUBIC, 
                                        borderMode=cv2.BORDER_REFLECT_101)
    
    if src_cropped_warped.dtype != dst_img.dtype:
        src_cropped_warped = src_cropped_warped.astype(dst_img.dtype)
    
    dst_img_rect = dst_img[dst_rect[1]:dst_rect[1] + dst_rect[3], dst_rect[0]:dst_rect[0] + dst_rect[2]]
    dst_img_rect = cv2.add(dst_img_rect * (1 - dst_mask[..., None] / 255.0), src_cropped_warped * (dst_mask[..., None] / 255.0))
    dst_img[dst_rect[1]:dst_rect[1] + dst_rect[3], dst_rect[0]:dst_rect[0] + dst_rect[2]] = dst_img_rect


def transfer_expression(source_img, target_img, source_landmarks, target_landmarks):
    source_triangulation = apply_delaunay(source_img, source_landmarks)
    
    for tri_indices in source_triangulation.simplices:
        src_tri = [source_landmarks[tri_indices[0]], source_landmarks[tri_indices[1]], source_landmarks[tri_indices[2]]]
        dst_tri = [target_landmarks[tri_indices[0]], target_landmarks[tri_indices[1]], target_landmarks[tri_indices[2]]]
        warp_triangle(source_img, target_img, src_tri, dst_tri)
    
    return target_img


def save_2d_landmarks(image, landmarks, save_path, circle_size=5):
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), circle_size, (0, 0, 255), -1) 
    cv2.imwrite(save_path, image)

def save_3d_landmarks_html(landmarks_3d, save_path):
    fig = go.Figure(data=[go.Scatter3d(
        x=landmarks_3d[:, 0], y=landmarks_3d[:, 1], z=landmarks_3d[:, 2],
        mode='markers', marker=dict(size=3, color='blue', opacity=0.8),
    )])
    fig.update_layout(
        scene=dict(xaxis_title="X Axis", yaxis_title="Y Axis", zaxis_title="Z Axis", aspectratio=dict(x=1, y=1, z=1)),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    fig.write_html(save_path)

def compute_rmse(source_landmarks, target_landmarks):
    return np.sqrt(mean_squared_error(source_landmarks, target_landmarks))

def compute_ssim(source_img, target_img):
    source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(source_gray, target_gray, full=True)
    return score

def compute_mae_3d(source_landmarks_3d, target_landmarks_3d):
    return np.mean(np.abs(source_landmarks_3d - target_landmarks_3d))
