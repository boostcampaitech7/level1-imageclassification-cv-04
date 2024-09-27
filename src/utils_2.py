import torch
import torch.nn.functional as F

# 먼저 두 모델의 예측 결과가 올바른 형태인지 확인
print("Model 1 predictions shape:", predictions_model_1.shape)
print("Model 2 predictions shape:", predictions_model_2.shape)

# 데이터 타입과 장치(device) 일관성 확보
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictions_model_1 = torch.tensor(predictions_model_1, dtype=torch.float32, device=device)
predictions_model_2 = torch.tensor(predictions_model_2, dtype=torch.float32, device=device)

# Soft Voting 수행
soft_voting = (predictions_model_1 + predictions_model_2) / 2
voting_result = torch.argmax(soft_voting, dim=1)

# 결과 확인
print("Soft voting shape:", soft_voting.shape)
print("Voting result shape:", voting_result.shape)

# model_2의 결과와 비교
model_2_result = torch.argmax(predictions_model_2, dim=1)
print("Matches with model_2:", (voting_result == model_2_result).sum().item())
print("Total predictions:", len(voting_result))

# 차이가 있는 예측 확인
diff_indices = (voting_result != model_2_result).nonzero(as_tuple=True)[0]
print("Indices where predictions differ:", diff_indices)

if len(diff_indices) > 0:
    print("Sample of differing predictions:")
    for idx in diff_indices[:5]:  # 처음 5개의 다른 예측 출력
        print(f"Index {idx}:")
        print(f"  Model 1 probs: {predictions_model_1[idx]}")
        print(f"  Model 2 probs: {predictions_model_2[idx]}")
        print(f"  Soft voting probs: {soft_voting[idx]}")
        print(f"  Soft voting prediction: {voting_result[idx]}")
        print(f"  Model 2 prediction: {model_2_result[idx]}")