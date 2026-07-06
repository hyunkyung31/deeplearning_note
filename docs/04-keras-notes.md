# Keras API 핵심 개념 — 공책용 정리

> `20260702_keras.ipynb` 학습 노트 (Day 2)

---

## 1. 전체 뼈대: Keras 실무 파이프라인

```
[환경] TensorFlow + Keras import
         ↓
[모델 정의] Functional API / Sequential API
         ↓
[compile] optimizer + loss + metrics
         ↓
[fit] data + epochs + batch_size + validation_data + callbacks
         ↓
[평가/시각화] evaluate / predict / hist.history 그래프
         ↓
[고급] Dropout, Callbacks, TensorBoard, Optuna
         ↓
[배포 대비] model.save / load_model
```

**한 줄 요약:** Functional/Sequential로 모델 정의 → compile → fit → Callbacks로 과적합 방지 → `.keras` 저장.

---

## 2. Keras 모델 정의 2가지 (★★★)

| 방식 | 코드 패턴 | 언제 쓰나 |
|------|-----------|-----------|
| **Functional** | `Input` → 레이어 연결 → `Model(inputs, outputs)` | 중간 출력 필요, feature map, 복잡한 구조 |
| **Sequential** | `Sequential([Input, Dense, ...])` | 레이어가 한 줄로 쌓일 때 (대부분) |

### Functional API

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(3,))
outputs = Dense(2, activation='relu')(inputs)
model = Model(inputs=inputs, outputs=outputs)

# 파라미터 수: (3×2) + 2 = 8
model.count_params()
model.get_config()  # 구조 확인
```

### Sequential API

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential([
    Input(shape=(784,)),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax'),
])

# 가중치 확인
model.layers[0].weights  # [W, b]
```

---

## 3. compile 3요소 (★★★)

```python
model.compile(
    optimizer='adam',           # 가중치 업데이트 방법
    loss='sparse_categorical_crossentropy',  # 틀린 정도 측정
    metrics=['accuracy'],       # 사람이 보기 좋은 점수
)
```

| 역할 | 예시 |
|------|------|
| **optimizer** | SGD, Adam, RMSprop |
| **loss** | MSE(회귀), crossentropy(분류) |
| **metrics** | accuracy, mae |

---

## 4. 분류 loss 3종류 — 헷갈리면 안 됨 (★★★)

| y 형태 | loss | 마지막 activation |
|--------|------|-------------------|
| 원-핫 `[0,0,1,0,...]` | `categorical_crossentropy` | softmax |
| 정수 `[0, 2, 5, ...]` | `sparse_categorical_crossentropy` | softmax 또는 logits |
| 실수 (회귀) | `mse` / `mean_squared_error` | 없음 (linear) |

### 노트북에서 배운 교훈

| 조합 | MNIST val accuracy |
|------|-------------------|
| sigmoid + MSE (잘못됨) | ~88% |
| softmax + categorical_crossentropy (올바름) | ~95% |
| Dropout + sparse_categorical_crossentropy | ~98% |

**loss와 activation은 반드시 짝이 맞아야 한다.**

---

## 5. to_categorical (★★☆)

```python
from tensorflow.keras.utils import to_categorical

y = [5, 0, 4]
y_onehot = to_categorical(y, num_classes=10)
# [[0,0,0,0,0,1,0,0,0,0],
#  [1,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,1,0,0,0,0,0]]
```

정수 라벨 → 원-핫 변환. `categorical_crossentropy` 쓸 때 필요.

---

## 6. 파라미터 수 계산 (★★★)

```
Dense: (들어오는 × 나가는) + bias(나가는)

예: Dense(512), 입력 784 → 784×512 + 512 = 401,920
예: Dense(10), 입력 512   → 512×10 + 10 = 5,130
```

---

## 7. Dropout (★★☆)

```
학습 중: 뉴런 20%를 무작위로 끔 → 과적합 방지
추론 중: 모든 뉴런 사용 (자동 처리)
파라미터 수: 0 (가중치 없음)
```

```python
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax'),
])
```

---

## 8. Callbacks 4종 (★★★) — Django 배포 전 필수

| Callback | 역할 | monitor |
|----------|------|---------|
| **EarlyStopping** | val_loss 안 줄면 학습 중단 | `val_loss` or `val_mae` |
| **ModelCheckpoint** | 최고 성능 모델 저장 | `val_loss` |
| **ReduceLROnPlateau** | loss 정체되면 학습률 감소 | `val_mae` |
| **TensorBoard** | 학습 과정 시각화 | log_dir |

```python
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)

early_stop = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)
checkpointer = ModelCheckpoint(
    filepath='best_model.keras', save_best_only=True, monitor='val_loss'
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_mae', factor=0.5, patience=5, min_lr=1e-6
)

callbacks = [early_stop, checkpointer, reduce_lr]
model.fit(..., callbacks=callbacks)
```

---

## 9. hist.history (★★☆)

```python
hist = model.fit(...)
hist.history['loss']        # epoch별 train loss
hist.history['val_loss']    # epoch별 validation loss
hist.history['accuracy']    # epoch별 train accuracy
hist.history['val_accuracy']
```

---

## 10. 회귀 vs 분류 (★★☆)

| | 분류 (MNIST) | 회귀 (Insurance) |
|--|-------------|-----------------|
| y | 정수/원-핫 | 실수 (보험료 $) |
| loss | crossentropy | MSE |
| metrics | accuracy | MAE |
| 마지막층 | softmax / logits | activation=None |
| 평가 | accuracy % | MAE ($ 오차) |

### Insurance 회귀 개선 기법

```python
model = Sequential([
    Input(shape=(n_features,)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1),  # 회귀: activation 없음
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

---

## 11. Optuna 하이퍼파라미터 튜닝 (★☆ — 개념)

```python
import optuna

def objective(trial):
    n_units = trial.suggest_int('n_units', 32, 256)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    # KFold CV로 점수 매기기
    return val_score

study = optuna.create_study(direction='minimize', pruner=MedianPruner())
study.optimize(objective, n_trials=100)
print(study.best_params)
```

흐름: `suggest_*` → `objective(trial)` → `study.optimize()` → `best_params`

---

## 12. 코드 리뷰 — 주의할 점

| # | 내용 | 조치 |
|---|------|------|
| 1 | sigmoid + MSE로 분류 시도 | softmax + CE로 수정 |
| 2 | `ReduceLROnPlateau`의 `facotr` 오타 | `factor=0.5` |
| 3 | `input_shape` 경고 | `Input(shape=(784,))` 먼저 넣기 |
| 4 | Insurance MAE ~$3,815 | Dropout + BatchNorm + L2 + Callbacks로 개선 |
| 5 | Optuna 100 trials 오래 걸림 | n_trials=10~20으로 연습, 개념만 |

---

## 13. 공책 필기 vs 키보드 타이핑

### 공책 + 연필 (개념·수식·표)

1. **Keras 4단계** 흐름도: 정의 → compile → fit → predict
2. **분류 loss 3종류** 표 (y 형태 / loss / activation)
3. **파라미터 계산** 예시: `784×512+512`, `512×10+10`
4. **Callbacks 4종** 역할 + monitor 종류
5. **Dropout** 왜 쓰는지 (과적합 방지, ensemble 효과)
6. **회귀 vs 분류** 비교표
7. **hist.history** 키 이름 4개
8. **Optuna** 흐름: suggest → objective → optimize → best_params

### 키보드 타이핑 (muscle memory)

| 우선 | 연습 내용 | 목표 |
|------|-----------|------|
| 1 | Functional API: `Input → Dense → Model` | 3분 |
| 2 | Sequential MNIST: Flatten→Dense→Dropout→Dense(softmax) | 5분 |
| 3 | compile 3줄 (optimizer, loss, metrics) | 1분 |
| 4 | fit + validation_data + callbacks | 3분 |
| 5 | Callbacks 리스트 3개 (EarlyStopping, ModelCheckpoint, ReduceLR) | 3분 |
| 6 | Insurance 회귀: Input→Dense→Dropout→Dense→Dense(1) + mse/mae | 5분 |

### 타이핑 연습 골격 (MNIST + Dropout)

```python
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax'),
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(
    x_train, y_train, epochs=5,
    validation_data=(x_test, y_test),
    callbacks=[early_stop, checkpointer],
)
```

---

## 14. 체크리스트

- [ ] Functional vs Sequential 차이
- [ ] compile 3요소 (optimizer, loss, metrics)
- [ ] 분류 loss 3종류 짝 맞추기
- [ ] to_categorical 용도
- [ ] Dense 파라미터 수 계산
- [ ] Dropout 역할 (파라미터 0)
- [ ] Callbacks 4종 + monitor
- [ ] hist.history 키 이름
- [ ] 회귀 vs 분류 비교
- [ ] Optuna 흐름 (개념)
