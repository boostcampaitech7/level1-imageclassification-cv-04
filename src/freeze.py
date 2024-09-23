def freeze(model):
    '''
    모델의 일부 layer만을 학습하기 위해 세팅하는 함수입니다
    '''
    # 모델의 모든 파라미터 requires_grad를 False로 설정 (동결)
    for _, param in model.model.named_parameters():
        param.requires_grad = False

    # Classification head만 학습 가능하게 설정
        model.model.head.weight.requires_grad = True
        model.model.head.bias.requires_grad = True
    return model