import torch.nn as nn
def layer_modification(model):
    '''
    모델의 일부 layer만을 학습하거나, layer를 수정하기 위해 세팅하는 함수입니다
    '''
    # 모델의 모든 파라미터 requires_grad를 False로 설정 (동결)
    for _, param in model.model.named_parameters():
        param.requires_grad = False

    # Classification head만 학습 가능하게 설정
        model.model.head = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 500)
        )
        model.model.head.requires_grad = True
    return model