# 코드 리뷰 + 체크리스트 + 학습 방향

> `20260701_FFNN.ipynb` 학습 노트북 기준 피드백

---

## 1. 코드 리뷰 요약

### 잘한 점

1. **학습 난이도 곡선** — `matmul → Dense → GradientTape → 선형회귀 → Keras → MNIST 커스텀 루프` 순서가 이상적
2. **주석이 "왜"를 적으려는 방향** — bias 차원, activation 필요성, `from_logits=True` 등
3. **MNIST 파이프라인** — reshape, 정규화, train/val/test 분리, tf.data, @tf.function까지 프로젝트 골격 완성

### 반드시 고칠 포인트

| # | 문제 | 해결 |
|---|------|------|
| 1 | `tf.data` 체이닝 SyntaxError | 괄호로 감싸기 (02-code-snippets.md 9번 참고) |
| 2 | Epoch 출력 수천 줄 반복 | validation 루프 **밖**에서 epoch당 1번만 print |
| 3 | DenseModel 주석 `65→10` | `64→10`으로 수정 |
| 4 | ReLU + GlorotUniform | ReLU는 **HeNormal** 초기화 권장 |
| 5 | `from_logits=True` | 모델 마지막에 softmax **없어야** 함 |

---

## 2. 100% 완벽 이해 체크리스트

### A. 텐서 & 변수 (★★★)

- [ ] `tf.constant` vs `tf.Variable` 차이를 말로 설명할 수 있다
- [ ] shape `(batch, features)`를 읽을 수 있다
- [ ] MNIST `(60000,28,28) → (60000,784)` 변환 이유를 안다

### B. FFNN 한 층 (★★★)

- [ ] `Y = activation(X @ W + b)` 수식을 쓸 수 있다
- [ ] W shape = `(들어오는, 나가는)` 을 계산할 수 있다
- [ ] bias 개수 = 나가는 차수(units) 임을 안다
- [ ] activation 없으면 층 100개 = 1층과 같다는 것을 안다

### C. 역전파 4줄 (★★★)

- [ ] GradientTape → gradient → apply_gradients 순서를 코드 없이 쓸 수 있다
- [ ] `trainable_variables`가 W, b 전체를 모은 것임을 안다

### D. Keras 4단계 (★★★)

- [ ] 정의 → compile → fit → predict 순서
- [ ] compile에서 optimizer와 loss 역할

### E. 분류 손실함수 (★★★)

- [ ] Sparse vs Categorical 차이
- [ ] `from_logits=True` ↔ 모델 마지막 softmax 없음

### F. 데이터 파이프라인 (★★☆)

- [ ] train / val / test 역할 구분
- [ ] Dataset → shuffle → batch → prefetch

### G. @tf.function (★★☆)

- [ ] Eager vs Graph 차이
- [ ] 2차 호출이 빠른 이유 (그래프 캐싱)

---

## 3. 공책 vs 키보드 연습 분리

### 공책 + 연필 (개념·구조·차원)

1. FFNN 1층 수식 + shape 예시
2. 역전파 4줄 의사코드
3. Keras 4단계 흐름도
4. MNIST 전처리 순서
5. Sparse vs Categorical CE 차이
6. 초기화 규칙 (ReLU→He, sigmoid→Glorot)
7. 활성화 함수 그래프 스케치

**shape 연습 문제:**

> 입력 (100, 784), Dense(128) → Dense(64) → Dense(10)  
> 각 W, b shape?

<details>
<summary>정답</summary>

- W1: (784, 128), b1: (128,)
- W2: (128, 64), b2: (64,)
- W3: (64, 10), b3: (10,)

</details>

### 키보드 (직접 타이핑)

| 우선순위 | 연습 내용 | 목표 |
|----------|-----------|------|
| 1 | GradientTape + Dense(1) 1스텝 | 5분 안에 작성 |
| 2 | Sequential 선형회귀 4단계 | compile/fit/predict |
| 3 | tf.data 파이프라인 3줄 | 괄호 체이닝 포함 |
| 4 | DenseModel `__call__` | matmul→relu→logits |
| 5 | train_step + epoch 루프 | metric reset 위치 |

---

## 4. 학습 로드맵 (TensorFlow 1주)

| Day | 내용 |
|-----|------|
| Day 2 | Keras API + MNIST (Sequential) |
| Day 3 | Functional API + Callbacks (EarlyStopping) |
| Day 4 | Dropout, L2 + 과적합 실험 |
| Day 5 | model.save / load ← **Django 연동 대비** |
| Day 6 | Fashion-MNIST or CIFAR-10 미니 프로젝트 |
| Day 7 | 복습 + 7/15 프로젝트 주제 구상 |

---

## 5. MNIST FFNN 학습 결과 (참고)

- Training Accuracy: ~94.5%
- Validation Accuracy: ~94.8%
- Test Accuracy: ~94.7%
- Test Loss: ~0.18

FFNN 첫날 기준으로 충분히 좋은 결과.
