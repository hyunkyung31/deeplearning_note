# CNN 핵심 개념 — 공책용 정리

> `20260703_CNN(convolution_neural_network).ipynb` 학습 노트 (Day 3)

---

## 1. 전체 뼈대: CNN 파이프라인

```
[이미지 이해] 픽셀 = 숫자 행렬, 행×열, RGB/흑백
         ↓
[전통 CV] 가우시안 블러, Sobel 엣지 검출 (CNN이 자동으로 하는 것의 원리)
         ↓
[Conv1D] Iris 데이터로 Conv 개념 입문
         ↓
[Feature Map] Functional API로 중간 Conv 출력 추출·시각화
         ↓
[CNN MNIST] Conv2D → MaxPooling → Flatten → Dense
         ↓
[CNN + Dropout + Callbacks] ModelCheckpoint, EarlyStopping
         ↓
[저장/로드/이어학습] load_model → fit() 다시
```

**한 줄 요약:** 이미지 = 행렬 → 필터(커널)로 특징 추출 → Pooling으로 축소 → Flatten → Dense 분류.

---

## 2. FFNN vs CNN (★★★)

| | FFNN | CNN |
|--|------|-----|
| 입력 | 1D 벡터 (784,) | 2D/3D (28, 28, 1) |
| 연산 | 전체 연결 (Dense) | **지역 연결 (Conv)** |
| 특징 | 전역 패턴 | **공간적 패턴** (모서리, 곡선) |
| 파라미터 | 많음 (784×512) | **적음** (3×3 필터 공유) |
| MNIST 성능 | ~94.7% | **~99.3%** |

---

## 3. 이미지 shape (★★★)

```
(batch, height, width, channels)
  │       │       │        │
  │       │       │        └── 1=흑백, 3=RGB
  │       │       └── 가로 (열)
  │       └── 세로 (행)
  └── 한 번에 넣는 장수

MNIST: (60000, 28, 28) → reshape → (60000, 28, 28, 1)
```

**주의:** 이미지 `(600, 800)` = 가로 800 × 세로 600이지만, **행렬은 (600, 800)** = 600행 × 800열.

```python
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
# 또는
x_train = np.expand_dims(x_train, axis=-1)
```

---

## 4. Conv2D (★★★)

```python
Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
```

| 파라미터 | 의미 |
|----------|------|
| **filters** | 필터(커널) 개수 = 출력 채널 수 |
| **kernel_size** | 필터 크기 (3×3이 표준) |
| **padding='same'** | 출력 크기 = 입력 크기 (zero padding) |
| **padding='valid'** | padding 없음 → 크기 줄어듦 |

### 출력 크기 공식 (valid, stride=1)

```
출력 = (입력 - kernel + 1)

예: (28 - 3 + 1) = 26  →  (26, 26, 32)
```

**same padding이면:** 입력과 출력 크기 동일 (28×28).

---

## 5. MaxPooling2D (★★★)

```
2×2 영역에서 최댓값 1개 선택 → 크기 절반
(26, 26, 32) → MaxPooling(2, 2) → (13, 13, 32)
```

- 다운샘플링 (크기 줄이기)
- 위치 변화에 강건 (translation invariance)
- **파라미터 0** (학습 없음)

---

## 6. CNN 전체 파이프라인 + shape 변화 (★★★)

```
입력 (28, 28, 1)
  → Conv2D(32, 3×3, valid) + ReLU     → (26, 26, 32)
  → MaxPooling(2, 2)                   → (13, 13, 32)
  → Conv2D(64, 3×3, valid) + ReLU     → (11, 11, 64)
  → MaxPooling(2, 2)                   → (5, 5, 64)
  → Flatten()                          → (1600,)
  → Dense(128) + ReLU                  → (128,)
  → Dropout(0.5)
  → Dense(10) + softmax                → (10,)
```

### shape 연습 (손으로 풀기)

> 입력 (28,28,1), Conv2D(32,3×3,valid), MaxPool(2), Conv2D(64,3×3,valid), MaxPool(2), Flatten

| 단계 | shape |
|------|-------|
| 입력 | (28, 28, 1) |
| Conv2D(32) | **(26, 26, 32)** |
| MaxPool | **(13, 13, 32)** |
| Conv2D(64) | **(11, 11, 64)** |
| MaxPool | **(5, 5, 64)** |
| Flatten | **(1600,)** |

---

## 7. Conv 파라미터 수 (★★★)

```
Conv2D: (kernel_h × kernel_w × input_channels + 1) × filters

예: Conv2D(32, (3,3)), 입력 채널 1:
    (3 × 3 × 1 + 1) × 32 = 320

예: Conv2D(64, (3,3)), 입력 채널 32:
    (3 × 3 × 32 + 1) × 64 = 18,496
```

---

## 8. Feature Map (★★☆)

- Conv 레이어 **출력** = feature map
- 필터마다 다른 특징 추출 (가장자리, 곡선, 패턴...)
- Functional API로 중간 출력 뽑기:

```python
from tensorflow.keras.models import Model

layer_outputs = [
    layer.output for layer in model.layers if 'conv' in layer.name
]
feature_map_model = Model(inputs=model.input, outputs=layer_outputs)
feature_maps = feature_map_model.predict(single_image)
```

---

## 9. Conv1D (Iris 입문용)

```python
# Iris: (150, 4) → (150, 4, 1)로 reshape
X = X.reshape(X.shape[0], X.shape[1], 1)

model = Sequential([
    Conv1D(64, kernel_size=2, activation='relu', input_shape=(4, 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax'),
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam', metrics=['accuracy'],
)
```

Conv1D는 1차원 시퀀스에 필터 적용. Conv2D의 1D 버전.

---

## 10. 모델 저장/로드/이어학습 (★★★) — Django 배포 직결

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 저장 (ModelCheckpoint가 자동)
checkpointer = ModelCheckpoint('best_model.keras', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(..., callbacks=[checkpointer, early_stop])

# 로드
loaded_model = load_model('best_model.keras')

# 이어 학습 (필요 시 compile)
loaded_model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
)
loaded_model.fit(X_train, Y_train, epochs=30, callbacks=[early_stop, checkpointer])
```

`.keras` 파일에 **구조 + 가중치 + compile 정보** 모두 포함.

---

## 11. Sparse vs Categorical (CNN 노트북)

| 노트북 | y 형태 | loss | 마지막층 |
|--------|--------|------|----------|
| Conv1D Iris | 정수 | `sparse_categorical_crossentropy` | softmax |
| CNN MNIST (후반) | 원-핫 | `categorical_crossentropy` | softmax |

---

## 12. 코드 리뷰 — 주의할 점

| # | 내용 | 조치 |
|---|------|------|
| 1 | `# 26x26x32x64` 주석 | shape는 `(26,26,32)` — 4D 아님 |
| 2 | `input_shape` 경고 | `Input(shape=(28,28,1))` 먼저 |
| 3 | Test Accuracy **99.29%** | CNN + Dropout + Callbacks — FFNN 94.7% 대비 큰 향상 |
| 4 | load 후 compile | `.keras`에 compile 정보 있으면 생략 가능, 에러 나면 compile |
| 5 | `evaluate()[1]` = accuracy | `[0]=loss, [1]=accuracy` |

---

## 13. 공책 필기 vs 키보드 타이핑

### 공책 + 연필 (개념·수식·표)

1. **FFNN vs CNN** 비교표
2. **이미지 shape** `(batch, H, W, C)` + MNIST 예시
3. **Conv2D 출력 크기 공식**: `(입력 - kernel + 1)` / same이면 동일
4. **MaxPooling**: 2×2 → 크기 절반, 파라미터 0
5. **CNN 파이프라인** shape 변화 전체 (28→26→13→11→5→1600→128→10)
6. **Conv 파라미터 수** 공식 + 예시 2개
7. **Feature Map** 개념 + Functional API로 추출
8. **모델 저장/로드** 3단계

### 키보드 타이핑 (muscle memory)

| 우선 | 연습 내용 | 목표 |
|------|-----------|------|
| 1 | MNIST 전처리: reshape + /255 + expand_dims | 2분 |
| 2 | CNN Sequential 6줄 (Conv→Pool→Conv→Pool→Flatten→Dense→Dense) | 5분 |
| 3 | compile (categorical_crossentropy + adam) | 1분 |
| 4 | Callbacks (ModelCheckpoint + EarlyStopping) + fit | 3분 |
| 5 | load_model + fit (이어 학습) | 3분 |
| 6 | Functional API feature map 추출 | 5분 |

### 타이핑 연습 골격 (CNN MNIST)

```python
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax'),
])
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam', metrics=['accuracy'],
)

checkpointer = ModelCheckpoint('best.keras', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=10)

model.fit(
    X_train, Y_train, epochs=30, batch_size=200,
    validation_data=(X_test, Y_test),
    callbacks=[checkpointer, early_stop],
)
```

---

## 14. Day 1~3 연결 — 큰 그림

```
Day 1 (FFNN)          Day 2 (Keras)           Day 3 (CNN)
─────────────         ─────────────           ─────────────
텐서/Variable    →    Functional/Sequential →  Conv2D/Pool
GradientTape     →    compile/fit/predict  →  Feature Map
수동 학습 루프    →    Callbacks/Dropout    →  save/load
MNIST 94.7%      →    MNIST 98% + Insurance → MNIST 99.3%
```

**7/15 프로젝트 + Django 배포를 위해 지금 반드시 손에 익혀야 할 것:**

1. `Sequential` 또는 `Functional`로 모델 정의
2. `compile` → `fit` → `evaluate` → `predict`
3. `ModelCheckpoint`로 `.keras` 저장
4. Django view에서 `load_model` → `predict`

---

## 15. 체크리스트

- [ ] FFNN vs CNN 차이
- [ ] 이미지 shape (batch, H, W, C)
- [ ] Conv2D 출력 크기 공식 (valid / same)
- [ ] MaxPooling 역할 (파라미터 0)
- [ ] CNN 파이프라인 shape 변화 손으로 계산
- [ ] Conv 파라미터 수 공식
- [ ] Feature Map + Functional API 추출
- [ ] load_model + 이어 학습
- [ ] sparse vs categorical crossentropy
