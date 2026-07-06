# Transfer Learning 핵심 개념 — 공책용 정리

> `20260706_transfer_learning.ipynb` 학습 노트 (Day 4~5)

---

## 1. 전체 뼈대: 이 노트북 한 바퀴

```
[배경] 왜 전이학습? + CNN 아키텍처 진화 (VGG → Xception → EfficientNet)
         ↓
[복습] MNIST CNN + GRAD-CAM (모델이 어디를 보고 판단했는지)
         ↓
[데이터 공급] ImageDataGenerator / tf.data / tfds / 파일 직접 로드
         ↓
[실전] Dogs vs Cats — 직접 CNN vs 전이학습 비교
         ↓
[전이학습 핵심] base_model(imagenet) + GAP + Dropout + Dense
         ↓
[Xception / EfficientNetB0 / EfficientNetV2S]
         ↓
[튜닝] GridSearchCV + scikeras (KerasClassifier)
```

**한 줄 요약:** ImageNet으로 미리 학습된 CNN을 **특징 추출기**로 쓰고, 맨 위에 **분류기(FFNN)**만 새로 학습한다.

---

## 2. 왜 전이학습? (★★★)

| 문제 | 전이학습 해결 |
|------|--------------|
| 데이터가 적음 (수천 장) | ImageNet 140만 장에서 배운 특징 재사용 |
| 학습 시간 오래 걸림 | base_model 가중치 고정 → 학습 파라미터 극소 |
| 처음부터 CNN 학습 어려움 | 엣지·텍스처·형태 등 저수준 특징 이미 학습됨 |

**전이학습 3가지 방법 (노트북 + 실무):**

| 방법 | `trainable` | 설명 |
|------|-------------|------|
| **Feature Extraction** | `False` | base 가중치 고정, 분류기만 학습 (노트북 메인) |
| **Fine-tuning** | 일부/전체 `True` | base 일부 레이어도 미세 조정 (데이터 많을 때) |
| **LoRA** | adapter만 학습 | 대형 모델(LLM)에서 주로 사용 (개념만) |

---

## 3. CNN 아키텍처 진화 — 외울 표 (★★★)

### vanishing gradient(기울기 소실) 해결 방향

| 방향 | 대표 모델 | 특징 |
|------|-----------|------|
| **Deeper** (깊게) | VGG16/32, ResNet, NasNet | 레이어 수 증가, ResNet은 skip connection |
| **Wider** (넓게) | GoogLeNet, Inception, Xception, MobileNet | 채널/분기 구조, Depthwise Separable Conv |
| **균형** | **EfficientNet** B0~B7 | depth + width + resolution 동시 스케일링 |

### Xception (노트북 사용)

- Inception에서 영감, **wider** 구조
- **Depthwise Separable Convolution**: 채널별로 convolution → 파라미터 절약
- 노트북: `keras.applications.Xception(weights='imagenet', include_top=False)`

### EfficientNet (노트북 사용)

- **B0** (224×224) ~ **B7** — 클수록 정확하지만 무거움
- B0: 4M params / Xception: 20M params → **EfficientNet이 가볍고 효율적**
- 파생: EfficientNetDet (객체 탐지)

### 컴퓨터 비전 4대 태스크 (★★☆)

| 태스크 | 출력 | 예 |
|--------|------|-----|
| **Classification** | 클래스 1개 | 개/고양이 |
| **Object Detection** | 클래스 + bounding box | YOLO |
| **Segmentation** | 픽셀 단위 마스크 | U-Net |
| **Object Tracking** | 프레임 간 객체 추적 | 영상 |

---

## 4. GRAD-CAM — 모델 해석 (★★★)

> "모델이 이미지 **어느 부분**을 보고 그런 결정을 내렸는지" 시각화

### 핵심 개념

| 용어 | 의미 |
|------|------|
| **Feature Map** | 마지막 Conv 레이어 출력 |
| **Gradient** | 예측 클래스 점수를 feature map으로 미분 |
| **GAP** | Global Average Pooling — 채널별 평균 → 중요도(weight) |
| **Heatmap** | feature map × 중요도 → 합산 → 0~1 정규화 |

### GRAD-CAM 5단계 (손으로 그릴 것)

```
1. grad_model = Model([input], [last_conv_output, model.output])
2. GradientTape → class_channel = predictions[:, pred_index]
3. grads = tape.gradient(class_channel, conv_outputs)
4. pooled_grads = reduce_mean(grads, axis=(0,1))  ← 채널별 중요도
5. heatmap = sum(pooled_grads × conv_outputs) → ReLU → normalize
```

### 시각화 3종

| 그림 | 내용 |
|------|------|
| Original | 원본 이미지 |
| Heatmap | 빨간/노란 = 중요 영역 |
| Superimposed | 원본 + heatmap 합성 (40% 투명) |

**OpenCV 포인트:** `cv2.COLORMAP_JET`, BGR→RGB 변환 (`cv2.cvtColor`)

---

## 5. 데이터 공급 3가지 방법 (★★★)

### ① ImageDataGenerator + flow (Keras 전통)

```python
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(
    X_train, y_train, batch_size=128,
)
model.fit(
    train_generator,
    steps_per_epoch=15,        # 한 epoch에 몇 batch
    epochs=30,
    validation_data=test_generator,
    validation_steps=5,
)
```

| 파라미터 | 의미 |
|----------|------|
| `rescale=1./255` | 픽셀 0~255 → 0~1 |
| `steps_per_epoch` | epoch당 batch 수 (데이터 전부 안 써도 됨) |
| `validation_steps` | validation batch 수 |

**데이터 증강 옵션 (개념):** `rotation_range`, `width_shift_range`, `horizontal_flip`, `zoom_range` 등

### ② tf.data / TensorFlow Datasets (tfds)

```python
import tensorflow_datasets as tfds

ds, info = tfds.load('mnist', split='train', shuffle_files=True, with_info=True)
# element_spec: {'image': (28,28,1), 'label': ()}

tfds.show_examples(ds, info)       # 샘플 시각화
tfds.as_dataframe(ds.take(4), info)  # DataFrame으로 확인
```

- `tfds.list_builders()` — 사용 가능한 데이터셋 목록 (수백 개)
- CatsVsDogs, cifar10, imagenet 등 포함

### ③ 파일에서 직접 로드 (Dogs vs Cats)

```python
import cv2, os
from tqdm import tqdm

IMG_SIZE = 100
PATH = '/content/train'

for file in tqdm(os.listdir(PATH)):
    if 'cat' in file:
        category = 1
    else:
        category = 0  # dog

    img = cv2.imread(os.path.join(PATH, file), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    x_train.append(img)
    y_train.append(category)

x_train = np.array(x_train).reshape(-1, 100, 100, 1)
x_train = x_train / 255.0
```

**라벨링:** 파일명에 `cat`/`dog` 포함 여부로 자동 분류

---

## 6. 전이학습 핵심 패턴 (★★★★) — 반드시 손으로 칠 것

### Feature Extraction 전체 구조

```
입력 (100, 100, 1) 흑백
    ↓
Concatenate ×3 → (100, 100, 3)  ← 흑백을 RGB로 변환
    ↓
Rescaling (전처리)              ← 모델마다 다름!
    ↓
base_model(x, training=False)   ← ImageNet 가중치, 학습 안 함
    ↓
GlobalAveragePooling2D()        ← (H,W,C) → (C,)
    ↓
Dropout(0.2)
    ↓
Dense(2, activation='softmax')  ← 새로 학습하는 분류기
```

### 코드 골격 (Xception 예시)

```python
from tensorflow import keras

base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(100, 100, 3),
    include_top=False,           # ★ ImageNet 1000클래스 분류기 제거
)
base_model.trainable = False     # ★ Feature Extraction

inputs = keras.Input(shape=(100, 100, 1))
x = keras.layers.Concatenate()([inputs, inputs, inputs])
x = keras.layers.Rescaling(scale=1/127.5, offset=-1)(x)  # Xception 전처리
x = base_model(x, training=False)   # ★ inference mode
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(2, activation='softmax')(x)

model = keras.Model(inputs, outputs)
```

### 핵심 파라미터 4개 (★★★)

| 파라미터 | 값 | 의미 |
|----------|-----|------|
| `weights='imagenet'` | ImageNet 사전학습 가중치 로드 |
| `include_top=False` | 마지막 FC 분류층 제거 (1000→내 클래스) |
| `base_model.trainable=False` | base 가중치 고정 |
| `training=False` | BatchNorm 등 inference 모드 |

### Trainable vs Non-trainable (노트북 수치)

| 모델 | Trainable | Non-trainable |
|------|-----------|---------------|
| Xception | **4,098** (16 KB) | 20,861,480 (79 MB) |
| EfficientNetB0 | **2,562** (10 KB) | 4,049,571 (15 MB) |

→ 전체 2000만 개 중 **4000개만** 학습 = 빠르고 과적합 적음

---

## 7. GlobalAveragePooling2D (GAP) (★★★)

```
입력: (3, 3, 2048)  ← Xception 출력 feature map
출력: (2048,)       ← 채널별 평균 1개

파라미터: 0
```

**Flatten vs GAP:**

| | Flatten | GAP |
|--|---------|-----|
| 출력 크기 | H×W×C (매우 큼) | C |
| 파라미터 | 0 | 0 |
| 과적합 | Flatten + Dense면 위험 | **전이학습 표준** |

GRAD-CAM에서도 GAP 개념 사용: gradient를 spatial axis로 평균 → 채널 중요도

---

## 8. 전처리 (Rescaling) — 모델마다 다름! (★★★)

| 모델 | 전처리 | 코드 |
|------|--------|------|
| **일반** | 0~1 정규화 | `Rescaling(1./255)` |
| **Xception** | -1~1 범위 | `Rescaling(scale=1/127.5, offset=-1)` |
| **EfficientNet** | 0~1 정규화 | `Rescaling(1./255)` |

**흑백 → RGB 변환:**

```python
# 방법: Concatenate로 1채널을 3번 복제
x = keras.layers.Concatenate()([inputs, inputs, inputs])
```

ImageNet 모델은 **3채널(RGB)** 입력이 기본 → 흑백 데이터는 반드시 변환

---

## 9. Loss / Activation 짝 맞추기 — Dogs vs Cats (★★★★)

### 이진 분류 (2클래스: dog/cat)

| y 형태 | loss | 마지막 activation | metrics |
|--------|------|-------------------|---------|
| 정수 `[0, 1, 0, ...]` | `sparse_categorical_crossentropy` | softmax | accuracy |
| 정수 `[0, 1, 0, ...]` | `BinaryCrossentropy()` | **sigmoid** (1 neuron) | BinaryAccuracy |
| 원-핫 `[[1,0],[0,1]]` | `categorical_crossentropy` | softmax | accuracy |

### 노트북에서 터진 실수들

| # | 실수 | 에러/결과 | 올바른 방법 |
|---|------|-----------|-------------|
| 1 | `BinaryCrossentropy(from_logits=True)` + `softmax` | 경고 + ~50% accuracy | `from_logits=False` 또는 activation 제거 |
| 2 | `to_categorical(y)` + `BinaryCrossentropy` | rank mismatch | sparse CE 또는 sigmoid 1 neuron |
| 3 | y가 `(None, 2, 2, 2)` shape | ValueError | `y.reshape(-1)` 후 to_categorical |

**정답 조합 (EfficientNet + GridSearch):**

```python
outputs = layers.Dense(2, activation='softmax')(x)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # y = 정수 0/1
    metrics=['accuracy'],
)
```

---

## 10. Fine-tuning vs Feature Extraction (★★★)

### Feature Extraction (노트북 기본)

```python
base_model.trainable = False
# → base 가중치 전부 고정
# → Dense 분류기만 학습
```

### Fine-tuning (데이터 충분할 때)

```python
base_model.trainable = True
# 또는
for layer in base_model.layers[:-30]:
    layer.trainable = False  # 앞쪽 레이어 고정, 뒤쪽만 학습

# ★ Fine-tuning 시 learning rate 작게!
optimizer = keras.optimizers.Adam(learning_rate=1e-5)
```

| | Feature Extraction | Fine-tuning |
|--|-------------------|-------------|
| trainable | 분류기만 | base 일부/전체 |
| learning rate | 1e-3 (보통) | **1e-5 ~ 1e-4** (작게) |
| 데이터 필요량 | 적음 (~1000+) | 많음 (~10000+) |
| 과적합 위험 | 낮음 | 높음 |

---

## 11. GridSearchCV + scikeras (★★☆)

Keras 모델을 sklearn API로 감싸서 하이퍼파라미터 탐색:

```python
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

def create_model(dropout_rate=0.2, dense_units=128, learning_rate=1e-3):
    base_model = keras.applications.EfficientNetV2S(
        weights='imagenet', include_top=False, input_shape=(100, 100, 3)
    )
    base_model.trainable = False
    # ... (전이학습 패턴 동일)
    return model

model_wrapper = KerasClassifier(
    model=create_model,
    epochs=10,
    batch_size=32,
    verbose=0,
)

param_grid = {
    'model__dropout_rate': [0.2, 0.3],
    'model__dense_units': [64, 128],
    'model__learning_rate': [1e-3, 1e-4],
}

grid = GridSearchCV(estimator=model_wrapper, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)  # y = 정수 라벨
print(grid_result.best_params_)
```

**주의:** `y_train`은 **정수** (sparse) — `to_categorical` 하지 않음

---

## 12. 과적합 판단 — loss 그래프 (★★☆)

```python
plt.plot(epochs, val_loss, label='Validation Loss')
plt.plot(epochs, loss, label='Training Loss')
```

| 패턴 | 의미 |
|------|------|
| train↓ val↓ 같이 감소 | **정상 학습** (일반화 OK) |
| train↓ val↑ | **과적합** (Dropout, EarlyStopping 필요) |
| train≈ val≈ 0.69, acc≈ 50% | **학습 실패** (loss/activation 짝 확인) |

---

## 13. 코드 리뷰 — 노트북 주의점

| # | 내용 | 조치 |
|---|------|------|
| 1 | BinaryCE(from_logits=True) + softmax | softmax 제거 또는 from_logits=False |
| 2 | y shape (None,2,2,2) 에러 | y 1D 정수로 reshape |
| 3 | EfficientNet ~50% (random) | y 라벨/sparse CE 확인 |
| 4 | Xception ~65% (낮음) | epoch 더, fine-tuning, 데이터 증강 |
| 5 | `scikit-learn==1.52` 오타 | `1.5.2` 등 유효 버전 |
| 6 | test.zip 라벨 없음 | test는 predict만, val은 train split |
| 7 | `training=False` 누락 | BatchNorm inference mode 필수 |

---

## 14. 공책 필기 vs 키보드 타이핑

### 공책 + 연필 (개념·수식·표)

1. **전이학습 3방법** 표 (Feature Extraction / Fine-tuning / LoRA)
2. **CNN 아키텍처 진화** (Deeper / Wider / EfficientNet)
3. **GRAD-CAM 5단계** 흐름 + GAP 개념
4. **전이학습 파이프라인** shape 변화 (1ch→3ch→base→GAP→Dense)
5. **include_top / trainable / training=False** 3개 차이
6. **Loss 3종류** 표 (sparse CE / categorical CE / Binary CE)
7. **전처리** Xception(-1~1) vs EfficientNet(0~1)
8. **Trainable vs Non-trainable** 파라미터 수 (Xception 4098 vs 20M)
9. **과적합 loss 그래프** 패턴 3가지
10. **컴퓨터 비전 4태스크** (classification / detection / segmentation / tracking)

### 키보드 타이핑 (muscle memory) — 우선순위

| 우선 | 연습 내용 | 목표 |
|------|-----------|------|
| 1 | **전이학습 전체 골격** (base→GAP→Dropout→Dense) | 5분 |
| 2 | GRAD-CAM `make_gradcam_heatmap` 함수 | 5분 |
| 3 | ImageDataGenerator + flow + fit | 3분 |
| 4 | Dogs vs Cats 파일 로드 (cv2 + tqdm) | 5분 |
| 5 | compile loss 짝 맞추기 (sparse CE + softmax) | 2분 |
| 6 | GridSearchCV + KerasClassifier | 5분 |

### 타이핑 연습 골격 ① — 전이학습 (EfficientNetB0)

```python
base_model = keras.applications.EfficientNetB0(
    weights='imagenet',
    input_shape=(100, 100, 3),
    include_top=False,
)
base_model.trainable = False

inputs = keras.Input(shape=(100, 100, 1))
x = keras.layers.Concatenate()([inputs, inputs, inputs])
x = keras.layers.Rescaling(1./255)(x)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(2, activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(X_train, y_train, epochs=10,
          validation_data=(X_test, y_test))
```

### 타이핑 연습 골격 ② — GRAD-CAM

```python
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    return heatmap.numpy()
```

### 타이핑 연습 골격 ③ — ImageDataGenerator

```python
train_datagen = ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow(X_train, y_train, batch_size=128)

model.fit(
    train_gen,
    steps_per_epoch=len(X_train) // 128,
    epochs=30,
    validation_data=(X_test, y_test),
)
```

---

## 15. Day 1~5 연결 — 큰 그림

```
Day 1 FFNN        → 텐서, 역전파, Keras 4단계
Day 2 Keras       → Functional/Sequential, Callbacks, loss 짝
Day 3 CNN         → Conv2D, Pooling, shape 계산, save/load
Day 4~5 Transfer  → ImageNet base + GAP + 분류기, GRAD-CAM, tfds
         ↓
7/15 프로젝트     → 전이학습 모델 + Django load_model + predict
```

**Django 배포 직결 코드:**

```python
# views.py
model = tf.keras.models.load_model('best_model.keras')

def predict_image(image_bytes):
    img = preprocess(image_bytes)  # resize, normalize, expand_dims
    pred = model.predict(img)
    return np.argmax(pred)
```

---

## 16. 체크리스트

- [ ] 전이학습이 왜 필요한지 (데이터 적음, 시간 단축)
- [ ] Feature Extraction vs Fine-tuning 차이
- [ ] `weights='imagenet'`, `include_top=False`, `trainable=False`
- [ ] `training=False` 왜 필요한지 (BatchNorm)
- [ ] GAP vs Flatten
- [ ] 흑백→RGB Concatenate 트릭
- [ ] 모델별 전처리 (Xception -1~1, EfficientNet 0~1)
- [ ] GRAD-CAM 5단계 + GAP 개념
- [ ] ImageDataGenerator flow + steps_per_epoch
- [ ] tfds.load / show_examples
- [ ] Dogs vs Cats 파일 로드 + 라벨링
- [ ] 이진 분류 loss 3종 (sparse CE / categorical CE / Binary CE)
- [ ] loss + activation 짝 맞추기 (from_logits 함정)
- [ ] Trainable params vs Non-trainable params
- [ ] GridSearchCV + KerasClassifier
- [ ] 과적합 loss 그래프 읽기
- [ ] CNN 아키텍처: VGG / Xception / EfficientNet 차이

---

## 17. 벼락치기 한 페이지 요약

```
전이학습 = ImageNet CNN(고정) + GAP + Dropout + Dense(학습)

base_model = Xception/EfficientNetB0(
    weights='imagenet', include_top=False, input_shape=(H,W,3))
base_model.trainable = False

입력(흑백) → Concat×3 → Rescaling → base(training=False)
         → GAP → Dropout → Dense(N, softmax)

compile: sparse_categorical_crossentropy (y=정수)
         또는 BinaryCrossentropy + sigmoid (y=0/1, 1 neuron)

GRAD-CAM: gradient × feature map → heatmap → 어디를 봤는지

데이터: ImageDataGenerator.flow / tfds.load / cv2 직접 로드
```
