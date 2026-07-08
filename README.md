# 딥러닝 학습 노트 (TensorFlow / Keras)

TensorFlow·Keras 학습 과정(FFNN → Keras API → CNN → Transfer Learning → AE/GAN)에서 정리한 핵심 개념, 코드, 학습 계획 모음입니다.

## 문서 목록

| 파일 | 내용 |
|------|------|
| [01-ffnn-core-concepts.md](./docs/01-ffnn-core-concepts.md) | Day 1 — FFNN 핵심 개념 + 전체 흐름도 |
| [02-code-snippets.md](./docs/02-code-snippets.md) | 실습용 Python 코드 블록 모음 |
| [03-code-review-checklist.md](./docs/03-code-review-checklist.md) | FFNN 노트북 코드 리뷰 + 체크리스트 |
| [04-keras-notes.md](./docs/04-keras-notes.md) | Day 2 — Keras API (Functional/Sequential, Callbacks, Optuna) |
| [05-cnn-notes.md](./docs/05-cnn-notes.md) | Day 3 — CNN (Conv2D, Pooling, Feature Map, save/load) |
| [06-transfer-learning-notes.md](./docs/06-transfer-learning-notes.md) | Day 4~5 — Transfer Learning (GRAD-CAM, Xception, EfficientNet, tfds) |
| [07-ae-gan-notes.md](./docs/07-ae-gan-notes.md) | Day 6 — Autoencoder (Denoising, Anomaly, GAN, 손코딩 템플릿) |
| [08-gan-class-based.md](./docs/08-gan-class-based.md) | Day 6 — GAN 함수→클래스 베이스 전환 + metric + save/load |
| [09-nlp-timeseries-attention-notes.md](./docs/09-nlp-timeseries-attention-notes.md) | Day 2 보충 — 시계열 · NLP · Attention · OpenCV 통합 필기 노트 |

## 학습 일정 (참고)

- **1주차:** TensorFlow / Keras 마스터
- **2주차:** PyTorch
- **7/15:** 딥러닝 핵심 프로젝트 시작 → Django 웹 연동·배포

## 예제 코드

| 경로 | 내용 |
|------|------|
| [examples/gan_class_based.py](./examples/gan_class_based.py) | MNIST DCGAN 클래스 베이스 전체 코드 |

## 로컬 실행 환경

- Python, TensorFlow 2.x, Keras 3.x
- Google Colab (GPU) + Cursor / VS Code 병행
