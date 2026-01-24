# Ollama Setup Guide

ACF Local Edition works best with [Ollama](https://ollama.ai) for fully offline code generation.

## Installation

### 1. Install Ollama

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai/download](https://ollama.ai/download)

### 2. Start Ollama

```bash
ollama serve
```

Keep this running in a terminal. Ollama runs on `http://localhost:11434` by default.

### 3. Pull Required Models

ACF uses different model sizes based on task complexity:

```bash
# Minimum - for 8GB VRAM / 16GB RAM
ollama pull qwen2.5-coder:7b

# Recommended - for 16GB VRAM / 32GB RAM
ollama pull qwen2.5-coder:14b

# Full routing - for 24GB+ VRAM / 64GB RAM
ollama pull qwen2.5-coder:7b    # cheap tasks
ollama pull yi-coder:9b          # medium tasks
ollama pull qwen2.5-coder:32b   # complex tasks
```

### 4. Verify Installation

```bash
# List installed models
ollama list

# Test a model
ollama run qwen2.5-coder:7b "Say hello"
```

## Configuration

Edit `config.toml` to match your installed models:

### Minimal Setup (8GB VRAM)

```toml
[llm]
backend = "ollama"
model_general = "qwen2.5-coder:7b"
model_code = "qwen2.5-coder:7b"

[routing]
enabled = false
```

### Recommended Setup (16GB VRAM)

```toml
[llm]
backend = "ollama"
model_general = "qwen2.5-coder:14b"
model_code = "qwen2.5-coder:14b"

[routing]
enabled = false
```

### Full Routing Setup (24GB+ VRAM)

```toml
[llm]
backend = "ollama"
model_general = "qwen2.5-coder:14b"
model_code = "qwen2.5-coder:14b"

[routing]
enabled = true
model_cheap = "qwen2.5-coder:7b"
model_medium = "yi-coder:9b"
model_premium = "qwen2.5-coder:32b"
```

## Troubleshooting

### "404 Not Found" Error

This means the model specified in `config.toml` is not installed.

```bash
# Check what models you have
ollama list

# Update config.toml to use an installed model
# OR pull the missing model
ollama pull qwen2.5-coder:14b
```

### "Connection Refused" Error

Ollama is not running.

```bash
# Start Ollama
ollama serve
```

### Slow Performance

- Use smaller models (7b instead of 14b)
- Close other applications to free RAM
- Check GPU utilization with `nvidia-smi` (NVIDIA) or Activity Monitor (macOS)

### Out of Memory

Your model is too large for your hardware.

```bash
# Use a smaller model
ollama pull qwen2.5-coder:7b

# Update config.toml
model_general = "qwen2.5-coder:7b"
model_code = "qwen2.5-coder:7b"
```

## Recommended Models

| Model | Size | VRAM | Use Case |
|-------|------|------|----------|
| `qwen2.5-coder:7b` | 4.7GB | 8GB | Fast, good for simple tasks |
| `qwen2.5-coder:14b` | 9GB | 16GB | Balanced, recommended default |
| `qwen2.5-coder:32b` | 19GB | 24GB+ | Best quality, complex tasks |
| `yi-coder:9b` | 5GB | 10GB | Good alternative for medium tasks |
| `deepseek-coder-v2:16b` | 8.9GB | 16GB | Strong for code understanding |

## Custom Ollama URL

If running Ollama on a different machine or port:

```toml
[llm]
backend = "ollama"
base_url = "http://192.168.1.100:11434"
```

Or via environment variable:

```bash
export OLLAMA_BASE_URL="http://192.168.1.100:11434"
acf run "Build an API"
```
