# 2026-07-07 AE / GAN 실습 필기 + 손코딩 정리

> **출처:** `20260707_AE_VAE_GAN.ipynb`  
> **제외:** CelebA Conv AE + 잠재벡터 속성 편집 (별도 정리 완료)  
> **참고:** 노트북 제목에 VAE가 있으나, **VAE 코드는 없음** (AE + GAN 중심)

---

## 0. 오늘 학습 흐름 한눈에

```
[기초] UpSampling / Conv2DTranspose
  ↓
[AE-1] MNIST Conv Autoencoder (Sequential)
  ↓
[AE 활용] Denoising → Anomaly Detection → Image Retrieval
  ↓
[AE-2] Functional API로 Encoder/Decoder 분리
  ↓
[AE-3] Fashion-MNIST Subclass Model (784→64→784)
  ↓
[Transfer] Encoder 특징 → Dense 분류 / RandomForest
  ↓
[GAN] Generator + Discriminator + 커스텀 train_step
```

---

## 1. 필기 핵심 (암기용)

### 1-1. UpSampling2D vs Conv2DTranspose

| | UpSampling2D | Conv2DTranspose |
|---|---|---|
| 동작 | 픽셀 **복제** (값 그대로 늘림) | **학습 가능한** 업샘플링 |
| 파라미터 | 없음 | Conv 커널 있음 |
| 용도 | 단순 확대 | Decoder, GAN Generator |

```python
# UpSampling: (2,2,1,3) → (4,4,1,3)  값 복제
y = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

# Conv2DTranspose: stride=2 → 공간 크기 2배
layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same')
```

**체크:** CNN 텐서는 항상 `(batch, H, W, C)` 4차원. Pooling/Conv로 **채널 수는 필터 수**로 결정.

---

### 1-2. Autoencoder 공통 원리

```
입력 x ──Encoder──▶ z (잠재벡터) ──Decoder──▶ x̂ (복원)
Loss: x와 x̂의 차이 최소화 (MSE 또는 BCE)
```

| 항목 | Conv AE (MNIST) | Dense AE (Fashion) |
|---|---|---|
| 입력 | 28×28×1 | 28×28 (784) |
| Encoder | Conv+Pool | Flatten → Dense(64) |
| Decoder | UpSample+Conv | Dense(784) → Reshape |
| Loss | binary_crossentropy | MSE |
| 출력 활성화 | sigmoid (0~1) | sigmoid |

**체크:** `fit(X, X)` — 입력=정답 (비지도 재구성 학습)

---

### 1-3. AE 3가지 활용 (시험 단골)

| 활용 | 원리 | 핵심 코드 |
|---|---|---|
| **Denoising** | 노이즈 입력 → 깨끗한 출력 학습 | `noisy = x + 0.4*noise`, `clip(0,1)` |
| **Anomaly Detection** | 정상만 학습 → 복원 오차(MSE) 큰 샘플 = 이상 | `mse = mean((x - x̂)²)` |
| **Image Retrieval** | Encoder 잠재벡터로 유사도 검색 | `L2 distance`, `argsort` |

**Anomaly Detection 필기 포인트**
- 정상 데이터 MSE ≈ 낮음
- 학습 분포 밖 데이터 MSE ≈ 높음
- 실무: 정상 MSE 분포로 **threshold** 설정

**Image Retrieval 필기 포인트**
- `latent_vectors = encoder.predict(X_test)`
- `distances = np.linalg.norm(latent_vectors - query, axis=1)`
- `closest = np.argsort(distances)[1:5]`  ← 자기 자신(0번) 제외

---

### 1-4. Functional API vs Sequential

```
Sequential: 한 줄씩 쌓기 (간단)
Functional: Input → 레이어 연결 → Model(input, output) (Encoder/Decoder 분리 가능)
```

**분리하는 이유:** Encoder만 따로 `predict` → 분류/검색/특징추출에 재사용

---

### 1-5. Subclass Model Autoencoder

```python
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Sequential([Flatten(), Dense(latent_dim, activation='relu')])
        self.decoder = Sequential([Dense(784, activation='sigmoid'), Reshape((28,28))])
    def call(self, x):
        return self.decoder(self.encoder(x))
```

**체크:** 784 → 64 = **12배 압축** (784×4byte ≈ 3KB → 64×4byte = 256byte)

---

### 1-6. Encoder → Downstream Task (Transfer)

```
이미지 → [학습된 Encoder] → 64차원 특징 → Dense 분류기 / RandomForest
```

| 방법 | 장점 |
|---|---|
| Encoder + Dense | End-to-end fine-tuning 가능 |
| Encoder + RF | 딥 특징 + 전통 ML, 해석 용이 |

**RF 전처리:** `features.reshape(N, -1)` — 2D `(샘플, 특징)` 필수

---

### 1-7. GAN 핵심 (★ 손코딩 필수)

#### 구조
```
Noise z ──Generator──▶ Fake Image ──┐
                                    ├── Discriminator ──▶ 0 or 1 (logit)
Real Image ─────────────────────────┘
```

#### Generator shape (MNIST DCGAN)
```
100 → Dense(7×7×256) → Reshape(7,7,256)
  → Conv2DTranspose 128, s=2  → 14×14
  → Conv2DTranspose 64, s=2   → 28×28
  → Conv2D(1, 7, tanh)        → 28×28×1  (-1~1)
```

#### Discriminator
```
28×28×1 → Conv2D 64, s=2 → 14×14
        → Conv2D 128, s=2 → 7×7
        → Flatten → Dense(1)  # logit, sigmoid 없음
```

#### 데이터 전처리 (Generator tanh와 맞춤)
```python
x = (x - 127.5) / 127.5   # 0~255 → -1~1
```

#### 손실함수
```python
bce = BinaryCrossentropy(from_logits=True)

# D: 진짜→0.9 (label smoothing), 가짜→0
d_loss = bce(0.9, real_out) + bce(0, fake_out)

# G: 가짜를 진짜(1)처럼 속이기
g_loss = bce(1, fake_out)
```

#### train_step 패턴 (★★★)
```python
@tf.function
def train_step(real_images):
    noise = tf.random.normal([batch_size, latent_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake = generator(noise, training=True)
        real_out = discriminator(real_images, training=True)
        fake_out = discriminator(fake, training=True)
        d_loss = discriminator_loss(real_out, fake_out)
        g_loss = generator_loss(fake_out)
    # D 먼저, G 나중 gradient 적용
    d_opt.apply_gradients(zip(d_grads, d.trainable_variables))
    g_opt.apply_gradients(zip(g_grads, g.trainable_variables))
```

#### GAN 안정화 팁 (필기)
| 기법 | 설정 |
|---|---|
| LeakyReLU | `0.2` (음수도 약하게 통과) |
| Dropout (D) | `0.3` |
| BatchNorm (G) | Dense/Conv 뒤 |
| Label smoothing | real label `0.9` (not 1.0) |
| Adam beta_1 | `0.5` (default 0.9 → GAN용 0.5) |
| Learning rate | `2e-4` |

#### 생성 이미지 시각화
```python
generated = generator(noise, training=False)
generated = (generated + 1) / 2.0   # tanh(-1~1) → imshow용(0~1)
```

---

## 2. 손코딩 연습 — 빈칸 채우기

### [연습 1] MNIST Conv Autoencoder Encoder 부분

```python
autoencoder = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(28,28,1)),  # 28×28
    MaxPooling2D(2, padding='same'),                                           # 14×14
    Conv2D(8, 3, padding='same', activation='relu'),
    MaxPooling2D(2, padding='same'),                                           # 7×7
    Conv2D(8, 3, strides=2, padding='same', activation='relu'),                # 4×4  ← stride로 축소
    # --- Decoder ---
    Conv2D(8, 3, padding='same', activation='relu'),
    UpSampling2D(),                                                            # 8×8
    Conv2D(8, 3, padding='same', activation='relu'),
    UpSampling2D(),                                                            # 16×16
    Conv2D(16, 3, activation='relu'),
    UpSampling2D(),                                                            # 32×32 → same padding으로 28 맞춤
    Conv2D(1, 3, padding='same', activation='sigmoid'),
])
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=128)
```

---

### [연습 2] Denoising + Anomaly Detection

```python
def denoise(autoencoder, X_test, noise_factor=0.4):
    noisy = X_test + noise_factor * np.random.normal(size=X_test.shape)
    noisy = np.clip(noisy, 0., 1.)
    return autoencoder.predict(noisy)

def reconstruction_mse(model, X):
    recon = model.predict(X)
    return np.mean(np.square(X - recon), axis=(1, 2, 3))  # 샘플별 MSE
```

---

### [연습 3] Functional API Encoder/Decoder 분리

```python
# Encoder
enc_in = Input(shape=(28, 28, 1))
x = Conv2D(16, 3, padding='same', activation='relu')(enc_in)
x = MaxPooling2D(2, padding='same')(x)
x = Conv2D(8, 3, padding='same', activation='relu')(x)
x = MaxPooling2D(2, padding='same')(x)
z = Flatten(name='latent_space')(x)                    # 7×7×8 = 392
encoder = Model(enc_in, z)

# Decoder
dec_in = Input(shape=(392,))
x = Reshape((7, 7, 8))(dec_in)
x = Conv2D(8, 3, padding='same', activation='relu')(x)
x = UpSampling2D()(x)
x = Conv2D(8, 3, padding='same', activation='relu')(x)
x = UpSampling2D()(x)
x = Conv2D(16, 3, padding='same', activation='relu')(x)
out = Conv2D(1, 3, padding='same', activation='sigmoid')(x)
decoder = Model(dec_in, out)

autoencoder = Model(enc_in, decoder(encoder(enc_in)))
```

---

### [연습 4] Subclass AE + 특징 추출

```python
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Sequential([
            Flatten(),
            Dense(latent_dim, activation='relu'),
        ])
        self.decoder = Sequential([
            Dense(784, activation='sigmoid'),
            Reshape((28, 28)),
        ])
    def call(self, x):
        return self.decoder(self.encoder(x))

ae = Autoencoder(64)
ae.compile(optimizer='adam', loss=losses.MeanSquaredError())
ae.fit(x_train, x_train, epochs=10)

encoded = ae.encoder(x_test).numpy()   # (N, 64)
```

---

### [연습 5] Encoder + 분류기

```python
classifier = Sequential([
    ae.encoder,                              # freeze 가능: ae.encoder.trainable = False
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax'),
])
classifier.compile(optimizer='adam',
                   loss=losses.SparseCategoricalCrossentropy(),
                   metrics=['accuracy'])
classifier.fit(x_train, y_train, epochs=30, batch_size=256)
```

---

### [연습 6] GAN Generator + Discriminator (핵심만)

```python
def build_generator(latent_dim):
    return Sequential([
        Input(shape=(latent_dim,)),
        Dense(7*7*256, use_bias=False),
        BatchNormalization(), LeakyReLU(0.2),
        Reshape((7, 7, 256)),
        Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False),
        BatchNormalization(), LeakyReLU(0.2),
        Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),
        BatchNormalization(), LeakyReLU(0.2),
        Conv2D(1, 7, padding='same', activation='tanh'),
    ])

def build_discriminator(img_shape):
    return Sequential([
        Input(shape=img_shape),
        Conv2D(64, 4, strides=2, padding='same'), LeakyReLU(0.2), Dropout(0.3),
        Conv2D(128, 4, strides=2, padding='same'), LeakyReLU(0.2), Dropout(0.3),
        Flatten(),
        Dense(1),   # logit
    ])
```

---

### [연습 7] GAN train_step (손코딩 최우선)

```python
bce = keras.losses.BinaryCrossentropy(from_logits=True)

def d_loss(real_out, fake_out):
    real_loss = bce(tf.ones_like(real_out) * 0.9, real_out)
    fake_loss = bce(tf.zeros_like(fake_out), fake_out)
    return real_loss + fake_loss

def g_loss(fake_out):
    return bce(tf.ones_like(fake_out), fake_out)

@tf.function
def train_step(real_images):
    bs = tf.shape(real_images)[0]
    noise = tf.random.normal([bs, latent_dim])
    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        fake = generator(noise, training=True)
        r_out = discriminator(real_images, training=True)
        f_out = discriminator(fake, training=True)
        dl = d_loss(r_out, f_out)
        gl = g_loss(f_out)
    d_opt.apply_gradients(zip(dt.gradient(dl, d.trainable_variables), d.trainable_variables))
    g_opt.apply_gradients(zip(gt.gradient(gl, g.trainable_variables), g.trainable_variables))
    return dl, gl
```

---

## 3. 비교표 (시험 직전 5분)

| | Autoencoder | GAN |
|---|---|---|
| 학습 방식 | 지도(자기 자신) | 적대적 (G vs D) |
| 목표 | 입력 복원 | 새 데이터 생성 |
| Loss | MSE / BCE | BCE (진짜/가짜) |
| 잠재공간 | Encoder가 추출 | Noise에서 시작 |
| 출력 범위 | sigmoid→0~1 | tanh→-1~1 |

| | Conv AE | Dense AE |
|---|---|---|
| 공간 정보 | 유지 (CNN) | 버림 (Flatten) |
| 이미지용 | ✅ | △ (작은 이미지) |
| 속도 | 느림 | 빠름 |

---

## 4. 자주 틀리는 포인트

1. **AE loss vs 출력 활성화:** 0~1 정규화 → `sigmoid` + `BCE` / MSE도 가능
2. **GAN 데이터 스케일:** Generator `tanh` ↔ 입력 `(x-127.5)/127.5`
3. **Discriminator 마지막:** `Dense(1)` + `from_logits=True` (sigmoid 레이어 X)
4. **GradientTape 2개:** D loss는 D 변수만, G loss는 G 변수만 gradient
5. **Image Retrieval:** `argsort` 결과 `[0]`은 자기 자신 → `[1:5]` 사용
6. **UpSampling vs Conv2DTranspose:** MNIST AE는 UpSampling, CelebA/GAN은 Conv2DTranspose

---

## 5. 과제 / 다음 학습

- [x] GAN 클래스 베이스 전환 → [08-gan-class-based.md](./08-gan-class-based.md), [examples/gan_class_based.py](../examples/gan_class_based.py)
- [ ] GAN 모델 `.keras` 저장 + noise 입력 → 이미지 생성 함수 작성 (예제 코드 포함)
- [ ] CelebA 속성 편집: **랜덤 방향 X** → `(z_smile - z_neutral)` 평균으로 attribute_axis 학습
- [ ] VAE: reparameterization trick + KL loss (노트북 미포함, 추가 학습)

---

## 6. CelebA Conv AE (간략 참조)

> 상세 설명은 별도 정리. 여기서는 흐름만.

```
64×64×3 → Conv(s=2)×3 → 8×8×128 → Flatten → Dense(256)
256 → Dense(8192) → Reshape(8,8,128) → Conv2DTranspose×3 → 64×64×3 (sigmoid)
Loss: MSE | latent 조작: z + α * direction
```

---

*작성일: 2026-07-07 | hyunkyun31*
