# Sketch image Classification competition

대회는 [upstage](https://stages.ai/)에서 비공개로 진행되었으며 데이터셋과 평가방식은 모두 upstage 제공 방식으로 이뤄져 있습니다.

## 대회 개요

스케치는 인간의 상상력과 개념 이해를 반영하는 추상적이고 단순화된 형태의 이미지입니다. 이러한 스케치 데이터는 색상, 질감, 세부적인 형태가 비교적 결여되어 있으며, 대신에 기본적인 형태와 구조에 초점을 맞춥니다. 이는 스케치가 실제 객체의 본질적 특징을 간결하게 표현하는데에 중점을 두고 있다는 점을 보여줍니다.

스케치 데이터의 특성을 이해하고 스케치 이미지를 통해 모델이 객체의 기본적인 형태와 구조를 학습하고 인식하도록 함으로써, 일반적인 이미지 데이터와의 차이점을 이해하고 또 다른 관점에 대한 모델 개발 역량을 높입니다.

이를 통해 실제 세계의 복잡하고 다양한 이미지 데이터에 대한 창의적인 접근방법과 처리 능력을 높일 수 있습니다. 또한, 스케치 데이터를 활용하는 인공지능 모델은 디지털 예술, 게임 개발, 교육 콘텐츠 생성 등 다양한 분야에서 응용될 수 있습니다.


## Data

[ImageNet Sketch 데이터셋](https://github.com/HaohanWang/ImageNet-Sketch) 중 upstage가 선정한 이미지 수량이 많은 상위 500개의 객체를 선정하여 총 25035개의 이미지 데이터를 활용합니다.

## Model 

단일 모델 기준 성능이 가장 좋았던 [eva02_large_patch14_448.mim_m38m_ft_in22k_in1k](https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k)을 base 모델로 사용했습니다.

## Usage

### Installation


1. Clone the repository:
   ```
   git clone https://github.com/your-username/sketch-image-classification.git
   cd sketch-image-classification
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Unzip data
   ```
   cd data
   tar -zxvf data.tar.gz
   ```

### Training

```
python train.py --traindata_dir ./data/train --traindata_info_file ./data/train.csv --save_result_path ./train_result --log_dir ./logs --val_split 0.2 --transform_type albumentations --batch_size 64 --model_type timm --model_name eva02_large_patch14_448.mim_m38m_ft_in22k_in1k --pretrained True --learning_rate 0.001 --epochs_per_lr_decay 2 --scheduler_gamma 0.1 --epochs 5
```

hyperparameter는 원하는 만큼 수정할 수 있습니다.

### Inference


```
python inference.py
```

## Project Structure
```
project_root/
│
├── data/
│   ├── train/
│   └── test/
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── models.py
│   ├── trainer.py
│   ├── layer_modification
│   └── utils.py
│
├── main.py
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```


----
## Experiment (~24.09.20)
### Experiment 규칙
<br>
Step1 .  반드시 Kanban에 실험 계획을 올려주세요
<br>
  Step2. 실험 후 실험 내용은 브랜치 명 "exp/{실험할 내용}"으로 깃에 올려주세요.<br>
 Step3.  실험 결과는 구글 시트에 기록해주세요

### Hyperparameter Tuning Experiment
- Write in [Google sheet](https://docs.google.com/spreadsheets/d/1tuTotQ_ALJQyJPzXt2NMeeyWfkm5csweRrYfWxnff8A/edit?usp=sharing)



### 브랜치 작성 규칙
1. main 브랜치는 건들지 말아주세요
2. feature 관련 브랜치명은 "feat/{구현할 내용}".
3. 각종 실험 관련 브랜치명은 "exp/{실험할내용}".
4. 수정 사항 관련 브랜치명은 "fix/{수정할 내용}"
   
