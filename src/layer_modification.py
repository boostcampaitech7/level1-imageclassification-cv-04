import torch.nn as nn

def layer_modification(model):
    '''
    모델의 일부 layer만을 학습하거나, layer를 수정하기 위해 세팅하는 함수입니다
    '''
    # 모델의 모든 파라미터 requires_grad를 False로 설정 (동결)
    for _, param in model.model.named_parameters():
        param.requires_grad = False

    # Classification head를 residual 구조로 수정
    model.model.head = ResidualHead(1024, 500)
    model.model.head.requires_grad = True
    return model

class ResidualHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 2048),
            nn.GELU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, in_features),
            nn.GELU()
        )
        self.block2 = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, in_features),
            nn.GELU()
        )
        self.final = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x + self.block1(x)
        x = x + self.block2(x)
        return self.final(x)