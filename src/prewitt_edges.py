import cv2
import numpy as np

def enhance_edges(image, weight=0.3):
    # 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Prewitt 엣지 검출
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    prewitt_x = cv2.filter2D(gray, -1, kernel_x)
    prewitt_y = cv2.filter2D(gray, -1, kernel_y)
    edges = np.sqrt(prewitt_x**2 + prewitt_y**2).astype(np.uint8)
    
    # 엣지를 3채널로 변환
    edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 원본 이미지와 엣지를 합성
    enhanced = cv2.addWeighted(image, 1, edges_3channel, weight, 0)
    
    return enhanced
