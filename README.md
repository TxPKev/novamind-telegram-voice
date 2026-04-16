# novamind-telegram-voice

Bidirectional voice bridge between Telegram (MTProto) and local audio pipelines.

Low-level integration of Telegram P2P voice calls with external PCM audio sources. Built on Telethon and ntgcalls.

**Status:** Development. Public release pending verification.

**Requirements:** Windows x86_64, NVIDIA GPU with CUDA, Python 3.10+.

## Install

    pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    pip install ntgcalls==2.1.0 --no-deps
    pip install "Telethon>=1.36.0" cryptg --no-deps

## Configuration

    cp config.example.json config.json
    # Edit config.json with your Telegram api_id and api_hash
    # Source: https://my.telegram.org/apps

## Run

    python main.py

---

NovaMind Studios — Niedergösgen, Switzerland  
ki27@ik.me  
https://txpkev.github.io/NOVAMINDSTUDIO
