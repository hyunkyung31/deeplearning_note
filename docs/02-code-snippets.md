# FFNN 실습용 Python 코드 모음

> Colab / Cursor에서 바로 복사해 실행할 수 있는 코드 블록

---

## 1. 환경 확인

```python
import tensorflow as tf
print(f"TensorFlow 버전: {tf.__version__}")
print("GPU 사용 가능:", tf.config.list_physical_devices("GPU"))
print("GPU 이름:", tf.test.gpu_device_name())
```

---

## 2. 텐서 기초

```python
import tensorflow as tf

# 상수 vs 변수
scalar = tf.constant(10)
vector = tf.constant([1, 2, 3])
matrix = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)

weight = tf.Variable(5.0)  # 학습 대상
weight.assign(7.0)
weight.assign_add(3.0)

print(weight.numpy())  # 확인용
```

---

## 3. 행렬곱 + bias (FFNN 한 층의 수학)

```python
import tensorflow as tf

# 입력: (batch=2, features=3)
X = tf.constant([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
], dtype=tf.float32)

# 가중치: (3, 2) — (들어오는, 나가는)
W = tf.constant([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
], dtype=tf.float32)

# 편향: (2,) — 나가는 차수와 동일
b = tf.constant([0.1, 0.2], dtype=tf.float32)

Y = tf.matmul(X, W) + b
print(Y.numpy())
```

---

## 4. Dense 레이어 (행렬곱 + bias를 레이어로)

```python
import tensorflow as tf

X = tf.constant([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
], dtype=tf.float32)

dense = tf.keras.layers.Dense(units=2, use_bias=True)
Y = dense(X)

W, b = dense.get_weights()
print("W shape:", W.shape)  # (3, 2)
print("b shape:", b.shape)  # (2,)
```

---

## 5. GradientTape — 1스텝 학습

```python
import tensorflow as tf

X = tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32)
y = tf.constant([[6.0], [7.0], [8.0]], dtype=tf.float32)

dense = tf.keras.layers.Dense(units=1, use_bias=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

with tf.GradientTape() as tape:
    y_pred = dense(X)
    loss = tf.reduce_mean(tf.square(y_pred - y))  # MSE

grads = tape.gradient(loss, dense.trainable_variables)
optimizer.apply_gradients(zip(grads, dense.trainable_variables))

print("업데이트된 bias:", dense.get_weights()[1])
```

---

## 6. 선형회귀 — 수동 학습 루프

```python
import tensorflow as tf
import numpy as np

np.random.seed(42)
X_train = tf.convert_to_tensor(np.linspace(-1, 1, 100), dtype=tf.float32)
y_train = tf.convert_to_tensor(
    2 * X_train.numpy() + 1 + np.random.normal(0, 0.1, size=100),
    dtype=tf.float32,
)

w = tf.Variable(tf.random.normal([1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias")
learning_rate = 0.1
epochs = 50

for epoch in range(1, epochs + 1):
    with tf.GradientTape() as tape:
        y_pred = w * X_train + b
        loss = tf.reduce_mean(tf.square(y_pred - y_train))

    dw, db = tape.gradient(loss, [w, b])
    w.assign_sub(learning_rate * dw)
    b.assign_sub(learning_rate * db)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d} | Loss: {loss.numpy():.4f} | w: {w.numpy()[0]:.4f} | b: {b.numpy()[0]:.4f}")
```

---

## 7. Keras Sequential — 4단계 (정의 → compile → fit → predict)

```python
import tensorflow as tf
import numpy as np

np.random.seed(42)
X_train = tf.convert_to_tensor(np.linspace(-1, 1, 100), dtype=tf.float32)
y_train = tf.convert_to_tensor(
    2 * X_train.numpy() + 1 + np.random.normal(0, 0.1, size=100),
    dtype=tf.float32,
)

# ① 정의
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1),
])

# ② compile
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss="mse",
)

# ③ fit
model.fit(X_train, y_train, epochs=50, verbose=0)

# ④ predict
weights, biases = model.layers[0].get_weights()
print(f"w={weights[0][0]:.4f}, b={biases[0]:.4f}")
print("예측:", model.predict(np.array([[2.0]]), verbose=0)[0][0])
```

---

## 8. MNIST — Keras Sequential (고수준 API)

```python
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10),  # logits — softmax 없음
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
)

model.evaluate(x_test, y_test, verbose=0)
```

---

## 9. MNIST — tf.data 파이프라인 (올바른 문법)

```python
import tensorflow as tf

BATCH_SIZE = 64
BUFFER_SIZE = 1024

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(buffer_size=BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
```

---

## 10. MNIST — 커스텀 모델 (tf.Module + 학습 루프)

```python
import tensorflow as tf

class DenseModel(tf.Module):
    def __init__(self, input_size=784, hidden_units=64, num_classes=10):
        super().__init__()
        initializer = tf.initializers.HeNormal()  # ReLU + HeNormal

        self.w1 = tf.Variable(initializer(shape=[input_size, hidden_units]), name="w1")
        self.b1 = tf.Variable(tf.zeros([hidden_units]), name="b1")
        self.w2 = tf.Variable(initializer(shape=[hidden_units, hidden_units]), name="w2")
        self.b2 = tf.Variable(tf.zeros([hidden_units]), name="b2")
        self.w_out = tf.Variable(initializer(shape=[hidden_units, num_classes]), name="w_out")
        self.b_out = tf.Variable(tf.zeros([num_classes]), name="b_out")

    @tf.function
    def __call__(self, x):
        h1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.w2) + self.b2)
        logits = tf.matmul(h2, self.w_out) + self.b_out
        return logits  # softmax 없음


model = DenseModel()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(from_logits=True)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = loss_fn(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_acc.update_state(y, logits)
    return loss


# epoch 루프 (출력은 epoch당 1번만)
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    train_acc.reset_state()
    for x_batch, y_batch in train_dataset:
        train_step(x_batch, y_batch)
    print(f"Epoch {epoch + 1} | Acc: {train_acc.result().numpy():.4f}")
```

---

## 11. 손실함수 비교 — Sparse vs Categorical

```python
import tensorflow as tf

# 정수 라벨 (원-핫 불필요)
y_true_sparse = tf.constant([[0], [2]], dtype=tf.int32)
y_pred = tf.constant([[0.9, 0.05, 0.05], [0.1, 0.2, 0.7]], dtype=tf.float32)

loss_sparse = tf.keras.losses.SparseCategoricalCrossentropy()(y_true_sparse, y_pred)

# 원-핫 라벨
y_true_onehot = [[1, 0, 0], [0, 0, 1]]
loss_cat = tf.keras.losses.CategoricalCrossentropy()(y_true_onehot, y_pred.numpy())

print("Sparse Loss:", loss_sparse.numpy())
print("Categorical Loss:", loss_cat.numpy())
```

---

## 12. @tf.function 속도 차이 확인

```python
import tensorflow as tf
import time

@tf.function
def graph_add(x, y):
    return x + y

# 1차: 그래프 생성 (느림)
start = time.time()
result1 = graph_add(tf.constant(10), tf.constant(5))
print(f"1차: {result1.numpy()} ({time.time() - start:.4f}s)")

# 2차: 캐시 재사용 (빠름)
start = time.time()
result2 = graph_add(tf.constant(20), tf.constant(10))
print(f"2차: {result2.numpy()} ({time.time() - start:.4f}s)")
```

---

## 13. 모델 저장 (Django 연동 대비)

```python
# 학습 후 저장
model.save("my_mnist_model.keras")

# Django view 등에서 로드
loaded_model = tf.keras.models.load_model("my_mnist_model.keras")
prediction = loaded_model.predict(single_image_batch)
```
