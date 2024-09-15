
### Project Structure
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
   
