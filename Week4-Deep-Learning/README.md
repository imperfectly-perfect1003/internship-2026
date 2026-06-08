# Week 4 — Deep Learning Fundamentals

## Deliverable: CIFAR-10 Image Classifier

**Final Test Accuracy: 86.07%** ✅ (Target: >85%)

### Architecture
- 3 Conv blocks: 32 → 64 → 128 filters
- BatchNormalization after each Conv layer
- MaxPooling + Dropout (0.2 → 0.3 → 0.4)
- Dense(512) → Dropout(0.5) → Softmax(10)
- Total parameters: 1,345,066

### Training Config
- Optimizer: Adam with ReduceLROnPlateau
- Loss: Categorical Crossentropy
- Batch size: 64
- Epochs: 50 (best at epoch 45)
- Data augmentation: flip, rotate, shift, zoom

### Results
| Metric | Value |
|---|---|
| Test Accuracy | **86.07%** |
| Test Loss | 0.4242 |
| Best Epoch | 45 |

### Per-class Accuracy
| Class | F1-Score |
|---|---|
| automobile | 0.93 |
| ship | 0.92 |
| horse | 0.90 |
| airplane | 0.88 |
| truck | 0.87 |
| deer | 0.85 |
| frog | 0.88 |
| bird | 0.82 |
| dog | 0.80 |
| cat | 0.74 |

### Key Observations
- Cat (74% F1) was the hardest class — commonly confused with dog. 
  This is a known CIFAR-10 challenge due to visual similarity.
- ReduceLROnPlateau triggered at epoch 27 (lr: 0.001 → 0.0005),
  which caused accuracy to jump from 82% → 86%
- BatchNormalization significantly stabilized training vs without it

### Files
- `cifar10_classifier.ipynb` — Full training notebook with all outputs
