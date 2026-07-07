# GAN 함수 베이스 → 클래스 베이스 전환

> **기준:** `20260707_AE_VAE_GAN.ipynb` MNIST DCGAN (함수 베이스)  
> **목표:** `build_generator` / `build_discriminator`는 유지하고, `class GAN(keras.Model)`로 학습 로직 캡슐화

---

## 1. 구조 한눈에

```
build_generator()     ──▶  generator (Sequential)
build_discriminator() ──▶  discriminator (Sequential)
                              │
                              ▼
                    class GAN(keras.Model)
                      - self.generator
                      - self.discriminator
                      - self.g_optimizer / self.d_optimizer
                      - self.d_loss_metric / self.g_loss_metric
                      - train_step()  ← 함수베이스 train_step 이동
                              │
                              ▼
                    gan.fit(dataset, epochs=50)
```

**GAN 클래스는 G/D 네트워크를 대체하지 않는다.** 학습 루프·optimizer·metric만 묶는 **오케스트레이터**.

---

## 2. 함수 베이스 ↔ 클래스 베이스 대응표

| 함수 베이스 | 클래스 베이스 |
|---|---|
| `generator = build_generator(...)` | `self.generator = generator` (생성자 인자) |
| `discriminator = build_discriminator(...)` | `self.discriminator = discriminator` |
| `g_optimizer = Adam(...)` | `self.g_optimizer = Adam(...)` |
| `d_optimizer = Adam(...)` | `self.d_optimizer = Adam(...)` |
| `bce = BinaryCrossentropy(...)` | `self.bce = BinaryCrossentropy(...)` |
| `def discriminator_loss(...)` | `self._discriminator_loss(...)` |
| `def generator_loss(...)` | `self._generator_loss(...)` |
| `@tf.function def train_step(...)` | `def train_step(self, data)` (Model 메서드) |
| `d_losses = []` + `append` | `self.d_loss_metric.update_state(d_loss)` |
| `tf.reduce_mean(d_losses)` | `self.d_loss_metric.result()` |
| epoch마다 `d_losses = []` | `fit()`이 epoch 끝에 `metric.reset_state()` |
| `for epoch ... for batch ... train_step()` | `gan.fit(dataset, epochs=50)` |

---

## 3. metric 작동 원리 (왜 4가지가 세트인가)

`Mean` metric만 선언한다고 자동 동작하지 **않음**. 아래 4가지가 함께 있어야 한다.

| 단계 | 코드 | 역할 |
|---|---|---|
| ① 생성 | `self.d_loss_metric = Mean(name="d_loss")` | 누적 평균 객체 |
| ② 등록 | `@property def metrics(self): return [...]` | epoch 끝 reset 대상 |
| ③ 기록 | `self.d_loss_metric.update_state(d_loss)` | `append`와 동일 |
| ④ 반환 | `return {"d_loss": self.d_loss_metric.result()}` | 로그/history 출력 |

```python
# Mean 내부 (개념)
# update_state → total += loss, count += 1
# result()      → total / count  (= reduce_mean)
# reset_state() → total=0, count=0  (= d_losses = [])
```

---

## 4. 전체 코드 (클래스 베이스)

### 4-1. G / D 빌드 함수 (함수 베이스와 동일)

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_generator(latent_dim):
    model = keras.Sequential(name="generator")
    model.add(layers.Input(shape=(latent_dim,)))
    model.add(layers.Dense(7 * 7 * 256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2D(1, 7, padding="same", activation="tanh"))
    return model


def build_discriminator(img_shape):
    model = keras.Sequential(name="discriminator")
    model.add(layers.Input(shape=img_shape))
    model.add(layers.Conv2D(64, 4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, 4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))  # logit
    return model
```

### 4-2. 데이터 로드 (함수 베이스와 동일)

```python
def load_data():
    (x_train, _), (_, _) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")
    x_train = (x_train - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(60000).batch(128, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return dataset
```

### 4-3. GAN 클래스 (★ 전환 핵심)

```python
class GAN(keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)
        self.g_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.d_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def compile(self, **kwargs):
        # GAN은 loss/optimizer를 train_step에서 직접 처리 → compile은 형식상 호출
        super().compile(**kwargs)

    def _discriminator_loss(self, real_output, fake_output):
        real_loss = self.bce(tf.ones_like(real_output) * 0.9, real_output)
        fake_loss = self.bce(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def _generator_loss(self, fake_output):
        return self.bce(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(noise, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)

            d_loss = self._discriminator_loss(real_output, fake_output)
            g_loss = self._generator_loss(fake_output)

        d_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = gen_tape.gradient(g_loss, self.generator.trainable_variables)

        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }
```

### 4-4. 학습 + 생성 (fit 사용)

```python
LATENT_DIM = 100
IMG_SHAPE = (28, 28, 1)

generator = build_generator(LATENT_DIM)
discriminator = build_discriminator(IMG_SHAPE)

gan = GAN(generator, discriminator, LATENT_DIM)
gan.compile()

dataset = load_data()
history = gan.fit(dataset, epochs=50, verbose=1)

# history.history['d_loss'], history.history['g_loss'] 에 epoch별 loss 저장
```

### 4-5. 이미지 생성 + 저장 (과제용)

```python
def generate_and_show(gan, epoch, latent_dim=100, examples=25, save_dir="gan_images"):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated = gan.generator(noise, training=False)
    generated = (generated + 1) / 2.0  # tanh(-1~1) → imshow(0~1)

    fig = plt.figure(figsize=(5, 5))
    for i in range(examples):
        plt.subplot(5, 5, i + 1)
        plt.imshow(generated[i, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"gan_mnist_{epoch:03d}.png"))
    plt.show()


def save_models(gan, prefix="gan_mnist"):
    # GAN 전체가 아니라 G/D 각각 저장 (실무·과제 표준)
    gan.generator.save(f"{prefix}_generator.keras")
    gan.discriminator.save(f"{prefix}_discriminator.keras")


def load_models(prefix="gan_mnist", latent_dim=100, img_shape=(28, 28, 1)):
    generator = keras.models.load_model(f"{prefix}_generator.keras")
    discriminator = keras.models.load_model(f"{prefix}_discriminator.keras")
    gan = GAN(generator, discriminator, latent_dim)
    gan.compile()
    return gan


def generate_from_noise(gan, noise):
    """noise: (N, latent_dim) numpy or tensor"""
    imgs = gan.generator(noise, training=False)
    return (imgs + 1) / 2.0
```

---

## 5. 함수 베이스 학습 루프 vs fit()

**함수 베이스 (Before):**
```python
for epoch in range(1, epochs + 1):
    d_losses, g_losses = [], []
    for real_images in dataset:
        d_loss, g_loss = train_step(real_images)
        d_losses.append(d_loss)
        g_losses.append(g_loss)
    print(f"D: {tf.reduce_mean(d_losses):.4f} | G: {tf.reduce_mean(g_losses):.4f}")
    if epoch % 5 == 0:
        show_images(epoch)
```

**클래스 베이스 (After):**
```python
history = gan.fit(dataset, epochs=50, verbose=1)

for epoch in range(1, 51):
    if epoch % 5 == 0 or epoch == 1:
        generate_and_show(gan, epoch)
```

---

## 6. 클래스 베이스 전환 체크리스트

- [ ] `build_generator` / `build_discriminator` **그대로** 유지
- [ ] `GAN(keras.Model)` 생성자에 G, D, `latent_dim` 전달
- [ ] optimizer·bce를 `self`에 저장 (전역 변수 제거)
- [ ] `train_step(self, data)`에서 Tape 2개 + gradient 대상 분리
- [ ] D loss → `discriminator.trainable_variables`만
- [ ] G loss → `generator.trainable_variables`만
- [ ] `d_loss_metric.update_state()` + `return {name: result()}`
- [ ] `@property metrics` 반환 리스트 등록
- [ ] `gan.compile()` 후 `gan.fit()`
- [ ] 저장은 `generator.save()` / `discriminator.save()` 각각

---

## 7. 자주 하는 실수

| 실수 | 결과 |
|---|---|
| metric만 선언하고 `update_state` 안 함 | loss가 0 또는 갱신 안 됨 |
| `train_step`에서 `return` 안 함 | verbose/history에 loss 안 찍힘 |
| D loss로 G weights까지 gradient | 학습 붕괴 |
| GAN 전체 `save()` | optimizer state 꼬임 → G/D 따로 저장 |
| `@property metrics` 누락 | epoch마다 metric reset 안 됨 |

---

*작성일: 2026-07-07 | hyunkyun31*
