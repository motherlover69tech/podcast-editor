## HuggingFace Token Setup

You need a free HuggingFace token for **speaker diarization** (who's talking).

Without it: transcription works fine, but words won't have speaker labels.

### Get a token (30 seconds)

1. Go to **https://huggingface.co/settings/tokens**
2. Click **"New token"**
3. Type: `Read` (read-only access)
4. Name it: `podcast-editor`
5. Click **Generate** → copy the token (starts with `hf_`)

### Use it

**Docker (recommended):**
Create a `.env` file next to `docker-compose.yml`:
```
HF_TOKEN=hf_yourtokenhere
```

Then `docker compose up -d` will pick it up.

**Without Docker:**
```bash
export HF_TOKEN=hf_yourtokenhere
```

### What it unlocks

| Feature | Without token | With token |
|---|---|---|
| Transcription (faster-whisper) | ✅ Works | ✅ Works |
| Speaker labels (pyannote) | ❌ No labels | ✅ "SPEAKER_00", "SPEAKER_01" |
| Quality | Same | Same |

The model is `pyannote/speaker-diarization-3.1` — state of the art, runs fine on CPU. First run downloads ~200 MB.
