# Technical Specifications

## Model Details

### Primary Model: Unbabel/TowerInstruct-7B-v0.1

- **Architecture**: Llama-based (Mistral variant)
- **Parameters**: ~7 Billion
- **Type**: Instruction-tuned translation model
- **Hidden size**: 4096
- **Layers**: 32
- **Context length**: 4096 tokens

## Storage

### Model Cache Location
```
/mnt/efs/rohingarg/.cache/huggingface/hub/models--Unbabel--TowerInstruct-7B-v0.1/
```

- **Size**: ~14 GB
- **Status**: Will download on first run (~10-15 minutes)

### Dataset Cache
```
/mnt/efs/rohingarg/.cache/huggingface/datasets/
```

- Konkani: 1.4 MB ✅ Downloaded
- Arabic: 1.1 MB ✅ Downloaded

## Memory Requirements

### Current Configuration (FP16, No Quantization)

| Component | Memory |
|-----------|--------|
| Model weights | ~14 GB |
| Activations | ~2-4 GB |
| KV cache | ~1-2 GB |
| **Total per run** | **~16-18 GB** |

### Your Hardware
- **GPUs**: 8x NVIDIA H100 80GB HBM3
- **Available per GPU**: 80 GB
- **Usage per run**: ~18 GB
- **Headroom**: ~62 GB ✅✅✅

**Verdict**: Excellent! No memory constraints.

## Quantization Status

### For Inference/Ablation: ❌ NO Quantization

**Configuration**:
```python
pipeline_model = transformers.pipeline(
    model=args.model,
    device=0,
    torch_dtype=torch.float16  # FP16 only, no quantization
)
```

**Why**: With your 8x H100 80GB GPUs, quantization is unnecessary. FP16 gives:
- ✅ Best translation quality
- ✅ Fast inference
- ✅ Simple setup
- ✅ Plenty of memory headroom

### For Fine-tuning: ✅ YES, 4-bit Quantization

**Configuration** (in `translation_finetuning.py`):
```python
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    load_in_4bit=True  # Saves memory for optimizer
)
```

**Why**: Fine-tuning needs extra memory for gradients and optimizer states.

## Performance Comparison

### CPU vs GPU

| Metric | GPU (H100) | CPU |
|--------|-----------|-----|
| Speed per translation | 2-5 sec | 100-250 sec |
| Memory | 18 GB VRAM | 14-28 GB RAM |
| **Full ablation time** | **4-6 hours** | **200-300 hours** |

**Recommendation**: Use GPU (already configured) ✅

### Quantization Impact on Quality

| Precision | BLEU Impact | Memory | Speed |
|-----------|------------|--------|-------|
| **FP16** (current) | Baseline | 18 GB | 1x |
| INT8 | -0.1 to -0.3 | 9 GB | 0.9x |
| INT4 | -0.3 to -1.0 | 5 GB | 0.8x |

**Your setup**: Keep FP16 for best quality ✅

## Inference Speed Estimates

### Per Translation
- **GPU (FP16)**: 2-5 seconds
- **Batch processing**: ~200-300 translations/hour

### Full Ablation Study
- **Single language, 6 k values**: 2-3 hours
- **Both languages**: 4-6 hours
- **Quick test (3 k values)**: ~1 hour

## Deployment Architecture

```
HuggingFace Hub (Remote)
    ↓ (download on first run)
Local Cache (~/.cache/huggingface/)
    ├── Models: ~14 GB
    └── Datasets: ~2.5 MB
    ↓
GPU Memory (H100 #0)
    ├── Model: 14 GB
    ├── Activations: 2-4 GB
    └── Total: ~18 GB / 80 GB
    ↓
Inference Pipeline
    └── Results saved to disk
```

## Model Loading Process

### First Run
1. Check cache → Not found
2. Download from HuggingFace (~14 GB)
3. Save to cache
4. Load to GPU
5. Ready for inference

**Time**: ~10-15 minutes

### Subsequent Runs
1. Load from cache
2. Load to GPU
3. Ready

**Time**: ~1-2 minutes

## Resource Monitoring

### Check GPU Usage
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Memory usage in Python
python -c "import torch; print(f'GPU Memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB')"
```

### Check Disk Space
```bash
# Cache directory
du -sh ~/.cache/huggingface/

# Free space
df -h /mnt/efs/rohingarg/
```

## Alternative Models (If Needed)

| Model | Parameters | Memory | Speed vs Current | Quality |
|-------|-----------|--------|------------------|---------|
| **Current: TowerInstruct-7B** | 7B | 18 GB | 1x | Excellent |
| NLLB-200-1.3B | 1.3B | 4 GB | 3x | Good |
| NLLB-200-600M | 600M | 3 GB | 5x | Moderate |

**Recommendation**: Keep TowerInstruct-7B. You have the hardware for it! ✅

## Summary

✅ **Model**: TowerInstruct-7B (7B params, ~14 GB)  
✅ **Precision**: FP16 (no quantization for inference)  
✅ **Memory per run**: ~18 GB  
✅ **Your GPUs**: 8x H100 80GB (perfect fit!)  
✅ **Speed**: 2-5 sec/translation  
✅ **Quality**: Best possible  
❌ **CPU**: Not feasible (50-100x slower)

**Your current setup is optimal. No changes needed!**

