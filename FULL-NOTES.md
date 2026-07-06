# 딥러닝 통합 학습 노트 (복사용)

> FFNN + Keras + CNN 핵심 개념 + 코드 + 체크리스트 통합본

---

# Part 1. 전체 흐름

```
[1] 데이터 준비 → [2] 모델 설계 → [3] compile → [4] fit(학습) → [5] evaluate/predict
```

**한 줄 요약:** 데이터 → 예측 → loss → gradient → 가중치 수정

---

# Part 2. FFNN 한 층

```
Y = activation( X @ W + b )

X: (batch, d_in)
W: (d_in, d_out)
b: (d_out,)
Y: (batch, d_out)
```

---

# Part 3. 역전파 4줄

```python
with tf.GradientTape() as tape:
    y_pred = model(x)
    loss = loss_fn(y, y_pred)

grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

---

# Part 4. Keras 4단계

```python
model = tf.keras.Sequential([...])          # ① 정의
model.compile(optimizer=..., loss=...)      # ② compile
model.fit(x_train, y_train, epochs=...)     # ③ fit
model.predict(x_test)                       # ④ predict
```

---

# Part 5. MNIST 전체 코드 (Keras)

```python
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=64)
model.evaluate(x_test, y_test)
```

---

# Part 6. 체크리스트

- [ ] constant vs Variable
- [ ] W shape = (들어오는, 나가는)
- [ ] 역전파 4줄
- [ ] Keras 4단계
- [ ] SparseCE + from_logits=True
- [ ] train/val/test 구분
- [ ] tf.data: shuffle → batch → prefetch

---

# Part 7. 벼락치기 요약

| 항목 | 내용 |
|------|------|
| 한 층 | Y = activation(X@W+b) |
| 학습 | Tape → gradient → apply_gradients |
| Keras | 정의→compile→fit→predict |
| 분류 | SparseCE(from_logits=True), softmax 없음 |
| 데이터 | train/val/test, Dataset pipeline |

---

# Part 8. Keras API (Day 2) 요약

```
Functional/Sequential → compile → fit → Callbacks → .keras 저장
```

| 항목 | 내용 |
|------|------|
| 모델 정의 | Functional (`Input→Dense→Model`) / Sequential |
| 분류 loss | sparse CE(정수 y) / categorical CE(원-핫 y) |
| Callbacks | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau |
| Dropout | 과적합 방지, 파라미터 0 |
| 회귀 | MSE + MAE, 마지막층 activation=None |

자세한 내용: [04-keras-notes.md](./docs/04-keras-notes.md)

---

# Part 9. CNN (Day 3) 요약

```
이미지 (H,W,C) → Conv2D → MaxPool → Flatten → Dense → softmax
```

| 항목 | 내용 |
|------|------|
| shape | (batch, height, width, channels) |
| Conv 출력 (valid) | (입력 - kernel + 1) |
| MaxPool | 2×2 → 크기 절반, 파라미터 0 |
| MNIST 성능 | FFNN 94.7% → CNN 99.3% |
| 배포 | ModelCheckpoint → load_model → predict |

자세한 내용: [05-cnn-notes.md](./docs/05-cnn-notes.md)

---

# Part 10. Day 1~5 체크리스트

- [ ] FFNN: Y = activation(X@W+b), 역전파 4줄, Keras 4단계
- [ ] Keras: Functional/Sequential, loss 짝 맞추기, Callbacks 3종
- [ ] CNN: Conv shape 계산, 파라미터 수, save/load
- [ ] Transfer: include_top/trainable/GAP, GRAD-CAM, loss 짝 (이진분류)

---

# Part 11. Transfer Learning (Day 4~5) 요약

```
ImageNet base(고정) → GAP → Dropout → Dense → softmax
```

| 항목 | 내용 |
|------|------|
| 방법 | Feature Extraction (trainable=False) / Fine-tuning |
| base | Xception, EfficientNetB0/V2S, weights='imagenet' |
| 전처리 | 흑백→Concat×3, Xception: -1~1, EfficientNet: 0~1 |
| 해석 | GRAD-CAM (gradient × feature map → heatmap) |
| 데이터 | ImageDataGenerator / tfds / cv2 직접 로드 |
| 튜닝 | GridSearchCV + scikeras KerasClassifier |

자세한 내용: [06-transfer-learning-notes.md](./docs/06-transfer-learning-notes.md)

---

자세한 내용은 `docs/` 폴더의 분리 문서를 참고하세요.
