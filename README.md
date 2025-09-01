# AutoDist

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-Supported-green.svg)

**자동으로 단일 GPU HuggingFace 모델을 분산 추론을 위한 텐서 병렬화 및 파이프라인 병렬화로 변환하는 도구**

## Description

AutoDist는 단일 GPU HuggingFace 모델을 효율적인 추론을 위해 텐서 병렬화(Tensor Parallel) 및 파이프라인 병렬화(Pipeline Parallel)를 사용한 분산 설정으로 자동 변환하는 도구입니다.

### 주요 기능

- **텐서 병렬화 (TP)**: 모델의 가중치를 여러 GPU에 분산하여 메모리 사용량을 줄이고 처리 속도를 향상
- **파이프라인 병렬화 (PP)**: 모델의 레이어를 여러 GPU에 분산하여 대용량 모델 처리 가능
- **CUDA Graph 캡처**: `ModelWrapper`를 통한 선택적 CUDA 그래프 캡처로 커널 실행 오버헤드 감소
- **간단한 CLI**: 단일 GPU, TP, PP 추론을 위한 직관적인 명령줄 인터페이스
- **자동 캐싱**: TP/PP 샤드 및 스테이지 자동 생성 및 캐싱

## Requirements

- **Python**: 3.10 이상
- **PyTorch**: 2.7 이상
- **CUDA**: 호환 가능한 PyTorch 빌드가 포함된 CUDA 지원 GPU
- **기타 의존성**: `requirements.txt` 참조

### 하드웨어 요구사항

- **단일 GPU**: 최소 8GB VRAM (모델 크기에 따라 다름)
- **텐서 병렬화**: 2개 이상의 GPU (동일한 VRAM 권장)
- **파이프라인 병렬화**: 2개 이상의 GPU (다른 VRAM 크기 지원)

## Installation

### 1. 저장소 클론

```bash
git clone https://github.com/JaeminK/AutoDist.git
cd AutoDist
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 패키지 설치

```bash
pip install -e .
```

## Quick Start

### 단일 GPU 추론

```bash
python main.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --cache-dir /workspace/cache \
  --output-dir ./results \
  --seed 1234 \
  --min-output-length 1 \
  --max-output-length 512
```

### 텐서 병렬화 (2개 GPU, 2-way TP)

```bash
torchrun --nproc_per_node 2 main.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --cache-dir /workspace/cache \
  --output-dir ./results \
  --tensor-parallel-size 2 \
  --seed 1234 \
  --min-output-length 1 \
  --max-output-length 512
```

### 파이프라인 병렬화 (2개 GPU, 2 스테이지)

```bash
torchrun --nproc_per_node 2 main.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --cache-dir /workspace/cache \
  --output-dir ./results \
  --pipeline-parallel-size 2 \
  --seed 1234 \
  --min-output-length 1 \
  --max-output-length 512
```

### 텐서 + 파이프라인 병렬화 (4개 GPU, 2-way TP + 2 스테이지 PP)

```bash
torchrun --nproc_per_node 4 main.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --cache-dir /workspace/cache \
  --output-dir ./results \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2 \
  --seed 1234 \
  --min-output-length 1 \
  --max-output-length 512
```

## Benchmarks

벤치마크 스크립트를 사용하여 성능을 테스트할 수 있습니다:

```bash
# 단일 GPU 벤치마크
cd benchmarks
./test_single.sh

# 텐서 병렬화 벤치마크 (4 GPU)
./test_tp_dist.sh

# 파이프라인 병렬화 벤치마크 (2 GPU)
# 단, Micro-batch 가 적용되지 않았기 때문에 Throughput 개선은 볼 수 없음.
./test_pp_dist.sh

# 텐서 + 파이프라인 병렬화 벤치마크 (4 GPU)
./test_pp_tp_dist.sh
```


### 중요 사항

- TP/PP 실행 시 `world_size`는 `tensor_parallel_size * pipeline_parallel_size`와 같아야 합니다.
- `local_rank`는 TP와 PP 크기의 곱보다 작아야 합니다.
- `batch_size`는 단일 토큰 추론을 위해 반드시 `1`이어야 합니다.

## Advanced Usage

### 커스텀 모델 사용

```python
from autodist import ModelRunner, DistributedSampler

# 모델 러너 초기화
model_runner = ModelRunner(
    model_name_or_path="your/model/path",
    device="cuda:0",
    dtype=torch.bfloat16,
    tensor_parallel_size=2,
    pipeline_parallel_size=1,
    local_rank=0,
    world_size=2,
    capture_graph=True
)

# 추론 실행
output = model_runner.generate(input_ids, max_length=512)
```

### CUDA Graph 최적화

CUDA Graph 캡처를 활성화하여 커널 실행 오버헤드를 줄일 수 있습니다:

```python
model_runner = ModelRunner(
    # ... 기타 옵션들
    capture_graph=True  # CUDA Graph 캡처 활성화
)
```

### 디버깅 팁

1. **CUDA 그래프 비활성화**: `capture_graph=False`로 설정
2. **단일 GPU 테스트**: 먼저 단일 GPU에서 모델이 작동하는지 확인
3. **로그 확인**: 환경 변수 `CUDA_LAUNCH_BLOCKING=1` 설정


## Acknowledgments

- [HuggingFace Transformers](https://github.com/huggingface/transformers) - 모델 로딩 및 추론
- [PyTorch](https://pytorch.org/) - 딥러닝 프레임워크
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - 분산 추론 아이디어

---

<div align="center">

⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요!

</div>
