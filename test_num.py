import torch
import numpy as np
import pandas as pd

# 예시 텐서 생성 (실제 사용 시 이 부분은 제외하고 기존 텐서를 사용하세요)
data = torch.randn(10014, 500)

# 텐서를 NumPy 배열로 변환
numpy_array = data.numpy()

# NumPy 배열을 pandas DataFrame으로 변환
np.save('testSave.npy', data)

print("텐서가 성공적으로 CSV 파일로 저장되었습니다.")