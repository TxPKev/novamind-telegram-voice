# ari-telegram-voice

**Live bidirectional Telegram voice calls with a fully local AI — no cloud, no API keys, no internet.**

XTTS-v2 · Whisper Large-v3 · ntgcalls v2.1.0 native P2P API · Windows x86_64

---

## What this is

A proof of concept showing something nobody has published before:

> A local AI — voice-cloned, running entirely on your GPU — that **picks up real Telegram 1-on-1 calls**, listens in real time, understands speech via Whisper, processes it through a local pipeline, and responds back in a cloned voice. **Live. Bidirectional. No cloud.**

The caller experience: you dial a regular Telegram number. The AI answers, listens, thinks, and talks back — in real time.

Built as the voice interface layer for **[AriNet](https://github.com/TxPKev/AriNet_deterministicAI)**, a fully deterministic, 100% offline AI assistant.

---

## Why this is genuinely novel

After exhaustive research across GitHub, PyPI, Reddit, Hacker News, and developer blogs, **zero public projects combine all four layers**:

| Layer | Prior art | Status |
|---|---|---|
| Telegram voice messages (async) | Many bots (Whisper + Coqui + LLM) | ✓ solved |
| Discord voice AI (bidirectional) | Discord-VC-LLM, Discord-Local-LLM-VoiceChat-Bot | ✓ solved for Discord |
| Telegram group call raw PCM | MarshalX/tgcalls (2021, dead) | ✓ worked, dead |
| **Telegram private call + raw PCM + local AI** | **Nobody** | **← this repo** |

The specific gap: `py-tgcalls` (the active high-level wrapper) does not expose private 1-on-1 calls or raw PCM callbacks in its stable release. `ntgcalls` v2.1.0 has the full P2P API at the C++ / native Python binding level — but no one has published a working example connecting it to a real AI pipeline.

This is that example.

---

## Honest status

| Component | Status |
|---|---|
| ntgcalls v2.1.0 P2P API (`create_p2p_call`, `connect_p2p`, `send_external_frame`, `on_frames`) | ✅ **Confirmed — live tested, bidirectional working** |
| Python binding signatures (pybind11 snake_case) | **Inferred from C++/Go — no .pyi stubs ship with the package** |
| Telethon signaling (DH exchange, AcceptCallRequest) | **Confirmed — matches MTProto spec** |
| `_build_rtc_servers()` — PhoneCall → ntgcalls RTCServer | **Inferred — field names match Telegram TL schema** |
| XTTS `inference_stream()` → 48kHz call audio | **Confirmed, ~200ms first chunk on RTX 3070** |
| Whisper Large-v3 via faster-whisper | **Confirmed, production-tested** |

**This is a PoC, not a finished product.** The Python API of ntgcalls is not officially documented. The code is built from the C++ source, Go bindings, and the Telegram TL schema. Run `pip install ntgcalls==2.1.0 --no-deps` and inspect with `dir(ntgcalls.NTgCalls())` to verify method names on your system before running.

---

## Architecture

```
Caller dials Ari's Telegram number (private 1-on-1 call)
    │
    ▼
Telethon UserClient — MTProto signaling layer
  UpdatePhoneCall → PhoneCallRequested
  → GetDhConfigRequest → compute g_b
  → AcceptCallRequest (DH exchange)
  → PhoneCall arrives (caller confirmed)
    │
    ▼
ntgcalls v2.1.0 — WebRTC/SRTP transport (native C++)
  create_p2p_call()
  init_exchange() + exchange_keys() — finalise DH
  set_stream_sources(MediaSource.EXTERNAL) — raw PCM mode
  connect_p2p(RTCServer list from PhoneCall endpoints)
    │
    ├─► on_frames() callback — inbound PCM from caller
    │       48kHz int16 → float32
    │       → SimpleVAD (energy-based, replaceable with silero-vad)
    │       → Whisper Large-v3 (resampled to 16kHz)
    │       → AriNet pipeline / your AI
    │       → XTTS-v2 inference_stream() → 24kHz chunks
    │       → resample 48kHz → push to OutboundAudioLoop
    │
    └─► send_external_frame() — outbound PCM to caller
            OutboundAudioLoop pulls from queue
            pushes 10ms frames (960 bytes @ 48kHz) at exact 10ms pacing
```

---

## Why Telethon alone is not enough

Telegram voice call **audio is WebRTC/SRTP — peer-to-peer between clients**. Telethon handles only the MTProto signaling (call accept, DH key exchange, connection endpoints). The audio packets never go through MTProto. To send and receive audio you need a WebRTC stack. `ntgcalls` IS that WebRTC stack. There is no shortcut.

This also means approaches like "intercept the SRTP stream directly with Python" require implementing SRTP decryption + WebRTC state machine yourself — ntgcalls already does this correctly in C++.

---

## Hardware & Software

**Tested configuration:**
- Windows 11 x86_64
- NVIDIA RTX 3070 8GB VRAM
- Python 3.11, CUDA 12.1, torch 2.5.1+cu121

**Minimum recommended:**
- NVIDIA GPU with 6GB+ VRAM (XTTS-v2 ~3GB + Whisper large-v3 ~3GB)
- Python 3.10 or 3.11 (3.13 confirmed to have ntgcalls wheels on PyPI)
- Windows 10/11 x86_64

**Linux:** ntgcalls has Linux support. Full stack not tested on Linux — PRs welcome.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/TxPKev/ari-telegram-voice.git
cd ari-telegram-voice
```

### 2. Install PyTorch first

```bash
# ALWAYS install torch before everything else.
# This prevents any other package from silently downgrading it to CPU.
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install ntgcalls

```bash
# --no-deps is MANDATORY.
# ntgcalls lists dependencies that would overwrite your torch CUDA build.
pip install ntgcalls==2.1.0 --no-deps
```

### 4. Verify the P2P API is available

```python
import ntgcalls
obj = ntgcalls.NTgCalls()
print([m for m in dir(obj) if not m.startswith('_')])
# Look for: create_p2p_call, connect_p2p, send_external_frame, on_frames, set_stream_sources
```

If `create_p2p_call` is not in the list, the wheel for your Python version may be an older build. Try Python 3.13 or build from source.

### 5. Install remaining dependencies

```bash
pip install "Telethon>=1.36.0" faster-whisper TTS numpy scipy cryptg --no-deps
```

### 6. Get Telegram API credentials

1. Go to [my.telegram.org/apps](https://my.telegram.org/apps)
2. Create an app → get `api_id` and `api_hash`
3. The account running this script must be a **user account** (not a bot). Ari needs her own Telegram number.

### 7. Configure

```bash
cp config.example.json config.json
# Edit config.json — api_id, api_hash, phone number
```

`config.json` is in `.gitignore`. Never commit it.

### 8. Add voice sample (optional but recommended)

```
voice/ari_voice_sample.wav   — 5–30s, clean speech, 22kHz+
```

Point `config.json` → `speaker_wav` to this path. XTTS-v2 clones the voice on first load.

### 9. Run

```bash
python ari_voice_call.py
```

First run: Telethon asks for phone + verification code (one-time). After that, the session is saved locally.

---

## Integrating your AI pipeline

Replace the stub in `AriPipeline.process()`:

```python
class AriPipeline:
    def process(self, text: str) -> str:
        # Your pipeline here:
        # from core.pipeline import run_pipeline
        # return run_pipeline(text)
        return f"Ich habe verstanden: {text}"   # stub
```

Everything else — call handling, DH exchange, VAD, STT, TTS, audio loop — is wired up.

---

## Key technical details

### ntgcalls v2.1.0 P2P call flow

ntgcalls v2.1.0 (released February 5, 2026) has two separate call creation paths. Group calls use `create_call(chat_id)`. Private calls use `create_p2p_call(chat_id)` followed by:

```
init_exchange(chat_id, dh_config, g_a_hash)
exchange_keys(chat_id, g_a_or_b, fingerprint)
connect_p2p(chat_id, rtc_servers, library_versions, p2p_allowed)
```

This matches the Telegram MTProto `PhoneCallProtocol` and `PhoneConnection` types exactly.

### Raw PCM mode

Setting `MediaSource.EXTERNAL` in `set_stream_sources()` enables raw byte access:
- **Outbound:** `send_external_frame(chat_id, StreamDevice.MICROPHONE, pcm_bytes, FrameData(...))`
- **Inbound:** `on_frames()` callback delivers `Frame` objects with raw PCM data

Frame format: PCM 16-bit signed little-endian, 48000 Hz, mono, 960 bytes = 480 samples (10ms).

### XTTS-v2 streaming

`inference_stream()` yields audio chunks as they are generated. First chunk in ~200ms on RTX 3070. This makes the call feel natural — Ari starts talking before the full response is synthesised.

### Audio resampling chain

```
ntgcalls in:    48kHz int16 mono
Whisper in:     16kHz float32 mono   →  resample via scipy.signal.resample_poly
XTTS out:       24kHz float32 mono   →  resample to 48kHz
ntgcalls out:   48kHz int16 mono
```

### Frame pacing

WebRTC expects exactly one 10ms frame (960 bytes = 480 int16 samples @ 48kHz) every 10ms. `OutboundAudioLoop` maintains this timing with `time.perf_counter()` to prevent jitter.

---

## VB-Cable vs ntgcalls

| | VB-Cable (workaround) | ntgcalls (this repo) |
|---|---|---|
| Requires Telegram Desktop running | ✓ | ✗ — headless |
| Programmatic audio control | ✗ | ✓ full raw PCM |
| Auto-answers calls | ✗ manual | ✓ fully automatic |
| Works on a server / headless | ✗ | ✓ |
| Stable API | ✓ | PoC — see honest status above |

The VB-Cable approach (routing desktop audio through a virtual cable into Telegram Desktop) works today and is a valid production shortcut. This repository solves the underlying problem at the proper level.

---

## Known risks and open questions

1. **ntgcalls Python binding names** — pybind11 generates snake_case from C++ but no `.pyi` stubs ship. Run the `dir()` check above before assuming method names.
2. **Frame pacing jitter** — `OutboundAudioLoop` uses `time.perf_counter()`. On Windows, timer resolution is ~0.5ms — adequate for 10ms frames.
3. **RTCServer field mapping** — `PhoneConnection` fields (`ip`, `ipv6`, `port`, `username`, `password`) are mapped to `ntgcalls.RTCServer`. Field names inferred from TL schema — verify if connection fails.
4. **VRAM budget** — XTTS-v2 (~3GB) + Whisper large-v3 (~3GB) = ~6GB. On 8GB VRAM leave ≥1GB headroom. Load models before the call starts to avoid cold-start delays.
5. **VAD quality** — Energy-based VAD is simple but fragile in noisy environments. Replace with [silero-vad](https://github.com/snakers4/silero-vad) for production use.

---

## Contributing

Issues and PRs welcome. Priority areas:

- Verified `ntgcalls.NTgCalls()` method signatures (`.pyi` stubs)
- `RTCServer` field name confirmation from live call
- Linux compatibility
- silero-vad integration
- Latency measurements from other GPU configs

---

## Project context

Extracted from **[AriNet](https://github.com/TxPKev/AriNet_deterministicAI)** — a fully deterministic, 100% offline AI assistant:

- No cloud, no API keys, no internet dependency
- Voice-cloned local AI with its own Telegram number
- Deterministic pipeline (seed-controlled, reproducible)
- Guardian Framework (patent-pending) for device state evaluation

This repo isolates only the Telegram voice call transport as a standalone, reproducible PoC.

---

## License

 GNU AFFERO GENERAL PUBLIC LICENSE 3.0
---

## Author

**Kevin Kachramanow** — NovaMind Studios
25 years CNC engineering · software architect · offline-first AI systems
[github.com/TxPKev](https://github.com/TxPKev) · Gösgen, Kanton Solothurn, Switzerland
