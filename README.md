# ari-telegram-voice

**Live bidirectional Telegram voice calls with a fully local AI — no cloud, no API keys, no internet.**

XTTS-v2 + Whisper Large-v3 + ntgcalls · Windows x86_64 · RTX 3070 tested

---

## What this is

This is a proof of concept showing something nobody has published before:

> A local AI (voice-cloned, running entirely on your GPU) that **picks up real Telegram calls**, listens to you in real time, understands what you say via Whisper, processes it through a local AI pipeline, and responds back in a cloned voice — **live, bidirectional, no cloud involved**.

The caller experience: you dial a regular Telegram number. The AI answers, listens, thinks, and talks back — in real time.

Built as the voice interface for **[AriNet](https://github.com/TxPKev/AriNet_deterministicAI)**, a fully deterministic, 100% offline AI assistant system.

---

## Why this matters

### The gap in the ecosystem

| What exists | What was missing |
|---|---|
| Telegram bots that send voice messages (pre-recorded OGG) | **Live call answering with a real AI** |
| VB-Cable workarounds (route desktop audio into call) | **Programmatic raw PCM in/out via ntgcalls** |
| XTTS demos in Jupyter notebooks | **XTTS streaming into a live phone call** |
| Whisper transcription from files | **Whisper doing real-time VAD + transcription from call audio** |
| pytgcalls for music bots | **pytgcalls for private 1-on-1 calls with raw PCM** |

No public repository combines XTTS-v2 + pytgcalls/ntgcalls + Whisper for live private calls. This is that repository.

---

## Architecture

```
Caller dials Ari's Telegram number
    │
    ▼
Telethon UserClient
UpdatePhoneCall event → auto-accept
    │
    ▼
ntgcalls / pytgcalls GroupCallRaw
    ├─► Inbound PCM (48kHz, 16-bit, mono)
    │       │
    │       ▼
    │   SimpleVAD (energy-based)
    │   accumulates chunks until silence detected
    │       │
    │       ▼
    │   Whisper Large-v3 (faster-whisper, int8, CUDA)
    │   resample 48kHz → 16kHz → transcribe
    │       │
    │       ▼
    │   AriNet Pipeline (or your AI pipeline)
    │   text in → response text out
    │       │
    │       ▼
    │   XTTS-v2 inference_stream()
    │   first audio chunk in ~200ms (RTX 3070)
    │   output: 24kHz float32 chunks
    │       │
    │       ▼
    │   Resample 24kHz → 48kHz
    │   push to outbound queue
    │
    └─► Outbound PCM (48kHz, 16-bit, mono)
        ntgcalls pulls from queue every 20ms
        caller hears the AI response in real time
```

---

## Hardware & Software Requirements

**Tested on:**
- Windows 11 x86_64
- ASUS ZenBook Pro Duo, NVIDIA RTX 3070 8GB VRAM
- Python 3.11, CUDA 12.1, torch 2.5.1+cu121

**Minimum recommended:**
- NVIDIA GPU with 6GB+ VRAM (for XTTS-v2 + Whisper simultaneously)
- Python 3.10+
- Windows 10/11 x86_64 (ntgcalls ships prebuilt Windows wheels)

**Linux:** ntgcalls has Linux support but is primarily tested on Windows. PRs welcome.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/TxPKev/ari-telegram-voice.git
cd ari-telegram-voice
```

### 2. Install dependencies

```bash
# Install PyTorch with CUDA first (prevents pip from downgrading it later)
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install everything else
pip install "pytgcalls[telethon]" faster-whisper TTS numpy scipy cryptg
```

> **Warning:** Never run `pip install llama-cpp-python` or other packages without `--no-deps` in the same environment — it can silently downgrade PyTorch to the CPU version and break CUDA.

### 3. Get Telegram API credentials

1. Go to [my.telegram.org/apps](https://my.telegram.org/apps)
2. Create an app — get your `api_id` and `api_hash`

### 4. Configure

```bash
cp config.example.json config.json
# Edit config.json with your api_id, api_hash, phone number
```

**`config.json` is in `.gitignore` — never commit it.**

### 5. Add a voice sample (optional but recommended)

Put a clean WAV file (5–30 seconds, 22kHz or higher, minimal background noise) of the target voice in:
```
voice/ari_voice_sample.wav
```

Point `config.json` → `speaker_wav` to this path. XTTS-v2 clones the voice on first load.

### 6. Run

```bash
python ari_voice_call.py
```

First run: Telethon asks for your phone number and a verification code (one-time auth).  
After that: the session is saved locally and Ari stays online.

---

## Integrating your own AI pipeline

The `AriPipeline` class in `ari_voice_call.py` is a stub:

```python
class AriPipeline:
    def process(self, text: str) -> str:
        # Replace with your actual pipeline
        return f"Ich habe verstanden: {text}"
```

Replace the body of `process()` with a call to your local LLM pipeline, rule engine, or whatever logic you want to run. The rest of the call handling, VAD, STT, TTS, and audio streaming is already wired up.

---

## How it compares to the VB-Cable approach

| | VB-Cable (Weg 2) | ntgcalls (this repo) |
|---|---|---|
| **Works today** | ✓ | ✓ (with this code) |
| **Requires Telegram Desktop** | ✓ | ✗ — runs headless |
| **Programmatic control** | ✗ | ✓ |
| **Works on a server** | ✗ | ✓ |
| **Auto-answers calls** | ✗ manual | ✓ fully automatic |
| **Custom audio processing** | Limited | Full raw PCM access |
| **Latency** | Depends on Telegram Desktop | ~200ms first TTS chunk |

The VB-Cable approach (routing desktop audio through a virtual cable into Telegram Desktop) works and is a valid workaround. This repository solves the underlying problem properly: programmatic raw PCM in/out, automatic call answering, no GUI required.

---

## Key technical details

### ntgcalls vs old pytgcalls

Old `pytgcalls` (v0.x) only supported group calls. `ntgcalls` (wrapped by `pytgcalls` v2+) adds:
- Private 1-on-1 call support
- Raw PCM in/out (`GroupCallRaw` equivalent)
- Prebuilt Windows x86_64 wheels — no compilation needed
- Active maintenance (2025)

### XTTS-v2 streaming

`inference_stream()` is the key. Instead of waiting for the full audio to be synthesised (3–8 seconds for a long sentence), it yields PCM chunks as they are generated. First chunk in ~200ms on an RTX 3070. This makes real-time voice feel natural rather than robotic.

### VAD design

This PoC uses a simple energy-based VAD (RMS threshold in dB). It works well in quiet environments. For production use, replace with `silero-vad` or `webrtcvad` for better noise robustness.

### Audio pipeline

```
ntgcalls in:  48kHz, int16, mono
Whisper in:   16kHz, float32, mono  →  resample with scipy.signal.resample_poly
XTTS out:     24kHz, float32, mono  →  resample to 48kHz
ntgcalls out: 48kHz, int16, mono
```

Resampling uses `scipy.signal.resample_poly` (polyphase filter, high quality, fast).

---

## Project context

This repository is a public PoC extracted from **AriNet** — a fully deterministic, 100% offline AI assistant system built by NovaMind Studios.

AriNet features:
- No cloud. No API keys. No internet dependency.
- Voice-cloned local AI with own Telegram number
- Deterministic pipeline (seed-controlled, reproducible)
- Guardian Framework (patent-pending) for device state evaluation

This repository isolates only the Telegram voice call integration as a standalone, reproducible PoC so others can build on it.

---

## Contributing

Issues and PRs welcome. Especially interested in:
- Linux compatibility improvements
- Silero-VAD integration (replace energy-based VAD)
- Group call support (currently only private 1-on-1 calls)
- Latency measurements from other GPU configs

---

## License

MIT — see [LICENSE](LICENSE)

---

## Author

**Kevin Kachramanow** — NovaMind Studios  
25 years CNC engineering · software architect · offline-first AI systems  
GitHub: [TxPKev](https://github.com/TxPKev)  
Location: Gösgen, Kanton Solothurn, Switzerland
