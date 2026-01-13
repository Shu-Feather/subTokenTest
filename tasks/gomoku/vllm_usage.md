Output example:

```
================================================================================
VERBOSE MODE - vLLM Request
================================================================================
Model: meta-llama/Llama-3.2-8B-Instruct
Board Size: 9x9

Full Prompt:
You are an expert at analyzing Gomoku (Five-in-a-Row) game boards...
[Full system prompt]

User: Please analyze this Gomoku board...
[Board representation]Output example:

```
================================================================================
VERBOSE MODE - vLLM Request
================================================================================
Model: meta-llama/Llama-3.2-8B-Instruct
Board Size: 9x9

Full Prompt:
You are an expert at analyzing Gomoku (Five-in-a-Row) game boards...
[Full system prompt]

User: Please analyze this Gomoku board...
[Board representation]# vLLM Usage Guide for Gomoku Benchmark

This guide explains how to use vLLM for high-performance local model inference in the Gomoku Benchmark.

## Why vLLM?

vLLM provides:
- **High Throughput**: Up to 24x faster than HuggingFace Transformers
- **Efficient Memory Usage**: PagedAttention reduces memory waste
- **Batch Processing**: Continuous batching for better GPU utilization
- **Easy Integration**: Compatible with OpenAI API

## Installation

### Option 1: Install vLLM with CUDA support (recommended)

```bash
pip install vllm
```

### Option 2: Install from source

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

## Usage Methods

### Method 1: Direct vLLM Inference (Recommended)

Use the `vllm` model type for direct inference:

```bash
# Run with Llama 3.2 8B
python main.py --models llama-3.2-8b --board-sizes 9 --test-counts 50

# Run with Qwen 2.5 7B
python main.py --models qwen2.5-7b --board-sizes 9 15 --test-counts 100

# Enable verbose mode to see prompts and responses
python main.py --models llama-3.2-8b --board-sizes 9 --test-counts 2 --verbose
```

### Method 2: vLLM Server (For Multiple Clients)

Start a vLLM server:

```bash
# Start server with Llama model
vllm serve meta-llama/Llama-3.2-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1

# Or with Qwen model
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

Then use the server in the benchmark:

```bash
python main.py --models vllm-server --board-sizes 9 --test-counts 50
```

## Supported Models

The benchmark includes pre-configured support for:

### Llama Models
- `llama-3.2-8b`: meta-llama/Llama-3.2-8B-Instruct
- `llama-3.1-8b`: meta-llama/Llama-3.1-8B-Instruct

### Qwen Models
- `qwen2.5-7b`: Qwen/Qwen2.5-7B-Instruct
- `qwen2.5-14b`: Qwen/Qwen2.5-14B-Instruct

## Adding Custom Models

### Edit config.py

```python
from config import DEFAULT_MODELS, ModelConfig

# Add your custom model
DEFAULT_MODELS["my-model"] = ModelConfig(
    model_name="path/to/your/model",  # HuggingFace model path or local path
    model_type="vllm",
    max_tokens=2000,
    temperature=0.1
)
```

### Or use configuration file

Create `custom_config.json`:

```json
{
  "models": ["my-custom-model"],
  "custom_models": {
    "my-custom-model": {
      "model_name": "path/to/your/model",
      "model_type": "vllm",
      "max_tokens": 2000,
      "temperature": 0.1,
      "gpu_memory_utilization": 0.9
    }
  },
  "board_sizes": [9, 15],
  "test_counts": [100]
}
```

Run with:

```bash
python main.py --config-file custom_config.json
```

## Verbose Mode

Enable verbose mode to see detailed prompt and response information:

```bash
python main.py --models llama-3.2-8b --board-sizes 9 --test-counts 2 --verbose
```

Output example:

```
================================================================================
VERBOSE MODE - vLLM Request
================================================================================
Model: meta-llama/Llama-3.2-8B-Instruct
Board Size: 9x9

Full Prompt:
[System and user prompts displayed here]

Model Response:
Let me analyze this board...
<answer>WHITE_WINS</answer>
================================================================================
```

## Performance Tips

### 1. GPU Memory Utilization

Control GPU memory usage:

```python
ModelConfig(
    model_name="meta-llama/Llama-3.2-8B-Instruct",
    model_type="vllm",
    gpu_memory_utilization=0.9  # Use 90% of GPU memory
)
```

### 2. Tensor Parallelism

For multi-GPU setups:

```bash
# vLLM automatically detects available GPUs
# For 2 GPUs:
vllm serve meta-llama/Llama-3.2-8B-Instruct --tensor-parallel-size 2

# For 4 GPUs:
vllm serve meta-llama/Llama-3.2-8B-Instruct --tensor-parallel-size 4
```

### 3. Quantization

Use quantized models for faster inference:

```bash
vllm serve meta-llama/Llama-3.2-8B-Instruct --quantization awq
```

## Troubleshooting

### Issue: Out of Memory

**Solution 1**: Reduce GPU memory utilization

```python
ModelConfig(
    model_name="your-model",
    model_type="vllm",
    gpu_memory_utilization=0.7  # Reduce from 0.9
)
```

**Solution 2**: Use a smaller model

```bash
# Use 7B model instead of 14B
python main.py --models qwen2.5-7b
```

### Issue: Model Not Found

**Solution**: Ensure the model is downloaded

```python
from huggingface_hub import snapshot_download

snapshot_download("meta-llama/Llama-3.2-8B-Instruct")
```

### Issue: vLLM Import Error

**Solution**: Reinstall vLLM

```bash
pip uninstall vllm
pip install vllm --no-cache-dir
```

## Comparison: vLLM vs Transformers

| Feature | vLLM | Transformers |
|---------|------|--------------|
| Speed | 10-24x faster | Baseline |
| Memory | Efficient (PagedAttention) | High usage |
| Batch Processing | Continuous batching | Simple batching |
| Setup | Requires vLLM install | Standard PyTorch |
| API Compatibility | OpenAI-like | Native |

## Advanced Configuration

### Custom Sampling Parameters

```python
from src.vllm_interface import VLLMInterface
from vllm import SamplingParams

# Create custom interface
interface = VLLMInterface(config)
interface.sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    max_tokens=2000,
    repetition_penalty=1.1
)
```

### Using Local Model Paths

```python
DEFAULT_MODELS["my-local-model"] = ModelConfig(
    model_name="/path/to/local/model",
    model_type="vllm",
    max_tokens=2000
)
```

## Example Workflows

### Quick Test

```bash
# Test with 2 cases to verify setup
python main.py --models llama-3.2-8b --board-sizes 9 --test-counts 2 --verbose
```

### Full Benchmark

```bash
# Run comprehensive benchmark
python main.py \
    --models llama-3.2-8b qwen2.5-7b \
    --board-sizes 9 15 19 \
    --test-counts 100 200 500
```

### Compare Models

```bash
# Compare multiple models
python main.py \
    --models llama-3.2-8b llama-3.1-8b qwen2.5-7b qwen2.5-14b \
    --board-sizes 15 \
    --test-counts 200
```

## Migration from Ollama/Transformers

If you were using Ollama or Transformers, switch to vLLM:

**Before (Ollama)**:
```bash
python main.py --models llama3.2 qwen2.5
```

**After (vLLM)**:
```bash
python main.py --models llama-3.2-8b qwen2.5-7b
```

Benefits:
- 10-20x faster inference
- Better memory efficiency
- No need to run separate Ollama server
- Direct Python API

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [Model Support List](https://docs.vllm.ai/en/latest/models/supported_models.html)