# LM Studio Setup Guide

ACF Local Edition also supports [LM Studio](https://lmstudio.ai) as an alternative to Ollama.

## Installation

### 1. Download LM Studio

Download from [lmstudio.ai](https://lmstudio.ai) for your platform:
- macOS (Apple Silicon or Intel)
- Windows
- Linux

### 2. Download a Model

1. Open LM Studio
2. Go to the **Discover** tab
3. Search for a coding model:
   - `qwen2.5-coder` (recommended)
   - `deepseek-coder-v2`
   - `codellama`
4. Download the appropriate size for your hardware

### 3. Start the Local Server

1. Go to the **Local Server** tab (left sidebar)
2. Select your downloaded model
3. Click **Start Server**
4. Note the server URL (default: `http://localhost:1234`)

## Configuration

Edit `config.toml` to use LM Studio:

```toml
[llm]
backend = "lmstudio"
base_url = "http://localhost:1234"
model_general = "local-model"  # LM Studio uses "local-model" as the model name
model_code = "local-model"

[routing]
enabled = false  # LM Studio runs one model at a time
```

## Recommended Models

Download from LM Studio's Discover tab:

| Model | Size | RAM | Quality |
|-------|------|-----|---------|
| Qwen2.5-Coder 7B Q4 | ~4GB | 8GB | Good |
| Qwen2.5-Coder 7B Q8 | ~8GB | 12GB | Better |
| Qwen2.5-Coder 14B Q4 | ~9GB | 16GB | Best balance |
| DeepSeek Coder V2 16B Q4 | ~10GB | 20GB | Strong |

**Tip:** Q4 quantization is faster and uses less memory. Q8 is higher quality but slower.

## Troubleshooting

### "Connection Refused" Error

LM Studio server is not running.

1. Open LM Studio
2. Go to **Local Server** tab
3. Click **Start Server**

### Slow Responses

- Use a smaller model or lower quantization (Q4 instead of Q8)
- Enable GPU acceleration in LM Studio settings
- Close other applications

### Model Not Loading

- Check you have enough RAM
- Try a smaller quantization variant
- Restart LM Studio

## LM Studio vs Ollama

| Feature | Ollama | LM Studio |
|---------|--------|-----------|
| CLI-first | Yes | No |
| Multiple models | Yes | One at a time |
| Auto-download | `ollama pull` | GUI download |
| Model routing | Supported | Not supported |
| Apple Silicon | Optimized | Optimized |
| NVIDIA GPU | Supported | Supported |

**Recommendation:** Use Ollama for ACF if you want model routing (cheap/medium/premium models). Use LM Studio if you prefer a GUI for model management.
