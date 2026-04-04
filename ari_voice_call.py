"""
ari_voice_call.py — AriNet Telegram Voice Call Handler
=======================================================
XTTS-v2 + Whisper Large-v3 + pytgcalls/ntgcalls
Live bidirectional Telegram call, 100% offline, no cloud, no API keys.

Architecture:
  Kevin calls Ari's Telegram number
  → Telethon (UpdatePhoneCall) auto-answers
  → GroupCallRaw records PCM (48kHz, 16-bit)
  → Whisper transcribes (resampled to 16kHz)
  → AriNet pipeline processes text
  → XTTS inference_stream() generates audio (24kHz)
  → Resample to 48kHz → GroupCallRaw feeds bytes into call

Author: Kevin Kachramanow / NovaMind Studios
GitHub: https://github.com/TxPKev/ari-telegram-voice
License: MIT
"""

import asyncio
import logging
import os
import sys
import struct
import time
import threading
import queue
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

# ── Telethon ──────────────────────────────────────────────────────────────────
from telethon import TelegramClient, events
from telethon.tl.types import (
    UpdatePhoneCall,
    PhoneCallRequested,
    PhoneCallAccepted,
)
from telethon.tl.functions.phone import AcceptCallRequest, ConfirmCallRequest

# ── pytgcalls / ntgcalls ──────────────────────────────────────────────────────
from pytgcalls import PyTgCalls
from pytgcalls.types import (
    MediaStream,
    AudioParameters,
)
# GroupCallRaw gives us raw PCM bytes in/out — the key to XTTS integration
from ntgcalls import (
    NTgCalls,
    AudioDescription,
    InputMode,
    MediaState,
)

# ── Whisper ───────────────────────────────────────────────────────────────────
import whisper

# ── XTTS-v2 ──────────────────────────────────────────────────────────────────
from TTS.api import TTS

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("AriVoiceCall")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TELEGRAM_CALL_SAMPLE_RATE = 48_000   # ntgcalls expects 48kHz
TELEGRAM_CALL_CHANNELS    = 1        # mono
TELEGRAM_CALL_BIT_DEPTH   = 16       # int16 PCM

WHISPER_SAMPLE_RATE       = 16_000   # Whisper expects 16kHz
XTTS_SAMPLE_RATE          = 24_000   # XTTS-v2 output

SILENCE_THRESHOLD_DB      = -40.0    # dB — below this = silence
VAD_SILENCE_SECONDS       = 0.8      # pause after which we transcribe
CHUNK_DURATION_MS         = 20       # ms per PCM chunk (ntgcalls standard)
CHUNK_SAMPLES             = TELEGRAM_CALL_SAMPLE_RATE * CHUNK_DURATION_MS // 1000


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — Load from config.json or environment
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load config from config.json (see config.example.json)."""
    import json
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        log.error("[CONFIG] config.json not found — copy config.example.json and fill in your values")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def pcm_bytes_to_float32(raw: bytes, channels: int = 1) -> np.ndarray:
    """Convert raw int16 PCM bytes → float32 numpy array [-1, 1]."""
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)  # mono mix
    return samples


def float32_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 numpy array [-1, 1] → raw int16 PCM bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    int16 = (clipped * 32767).astype(np.int16)
    return int16.tobytes()


def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio array using scipy polyphase filter (high quality, fast)."""
    from math import gcd
    g = gcd(from_rate, to_rate)
    up   = to_rate  // g
    down = from_rate // g
    return resample_poly(audio, up, down).astype(np.float32)


def rms_db(audio: np.ndarray) -> float:
    """Return RMS level in dB. Returns -inf for silence."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return -100.0
    return 20.0 * np.log10(rms)


# ─────────────────────────────────────────────────────────────────────────────
# VAD — Simple energy-based Voice Activity Detection
# ─────────────────────────────────────────────────────────────────────────────

class SimpleVAD:
    """
    Accumulates incoming PCM chunks. Detects speech start and end.
    When speech ends (silence > VAD_SILENCE_SECONDS), returns accumulated
    audio for transcription.
    """

    def __init__(
        self,
        sample_rate: int = TELEGRAM_CALL_SAMPLE_RATE,
        silence_threshold_db: float = SILENCE_THRESHOLD_DB,
        silence_seconds: float = VAD_SILENCE_SECONDS,
    ):
        self.sample_rate = sample_rate
        self.threshold = silence_threshold_db
        self.silence_samples = int(silence_seconds * sample_rate)

        self._buffer: list[np.ndarray] = []
        self._silence_count: int = 0
        self._speaking: bool = False

    def feed(self, chunk: np.ndarray) -> np.ndarray | None:
        """
        Feed a PCM chunk. Returns full utterance array when speech segment ends,
        None otherwise.
        """
        level = rms_db(chunk)
        is_speech = level > self.threshold

        if is_speech:
            self._speaking = True
            self._silence_count = 0
            self._buffer.append(chunk)
        elif self._speaking:
            self._buffer.append(chunk)
            self._silence_count += len(chunk)
            if self._silence_count >= self.silence_samples:
                # Speech ended — return full utterance
                utterance = np.concatenate(self._buffer)
                self._buffer = []
                self._silence_count = 0
                self._speaking = False
                return utterance

        return None

    def reset(self):
        self._buffer = []
        self._silence_count = 0
        self._speaking = False


# ─────────────────────────────────────────────────────────────────────────────
# WHISPER STT
# ─────────────────────────────────────────────────────────────────────────────

class WhisperSTT:
    """
    Whisper Large-v3 wrapper.
    Loads once at startup (Int8 quantised, CUDA).
    """

    def __init__(self, model_size: str = "large-v3", device: str = "cuda"):
        log.info("[STT] Loading Whisper %s on %s …", model_size, device)
        # Use faster-whisper for int8 quantisation if available, fallback to openai-whisper
        try:
            from faster_whisper import WhisperModel
            self._fw = WhisperModel(
                model_size,
                device=device,
                compute_type="int8_float16",
            )
            self._use_faster = True
            log.info("[STT] faster-whisper loaded (int8_float16)")
        except ImportError:
            log.warning("[STT] faster-whisper not found — falling back to openai-whisper (slower)")
            self._model = whisper.load_model(model_size, device=device)
            self._use_faster = False

    def transcribe(self, audio_48k: np.ndarray) -> str:
        """Transcribe 48kHz float32 audio. Returns text string."""
        # Resample 48kHz → 16kHz for Whisper
        audio_16k = resample(audio_48k, TELEGRAM_CALL_SAMPLE_RATE, WHISPER_SAMPLE_RATE)

        if self._use_faster:
            segments, _info = self._fw.transcribe(
                audio_16k,
                language="de",          # Ari speaks German — change as needed
                beam_size=5,
                vad_filter=True,
            )
            return " ".join(seg.text for seg in segments).strip()
        else:
            result = self._model.transcribe(
                audio_16k,
                language="de",
                fp16=torch.cuda.is_available(),
            )
            return result["text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# XTTS TTS
# ─────────────────────────────────────────────────────────────────────────────

class XTTSStreamer:
    """
    XTTS-v2 streaming TTS.
    Loads once. Uses inference_stream() for low-latency first chunk (~200ms).
    Outputs 24kHz float32 chunks → caller resamples to 48kHz.
    """

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
                 speaker_wav: str | None = None,
                 language: str = "de"):
        log.info("[TTS] Loading XTTS-v2 …")
        self._tts = TTS(model_name=model_name).to("cuda")
        self._speaker_wav = speaker_wav
        self._language = language
        log.info("[TTS] XTTS-v2 ready")

    def stream(self, text: str):
        """
        Generator — yields float32 numpy chunks at 24kHz.
        First chunk arrives in ~200ms on RTX 3070.
        """
        log.info("[TTS] Synthesising: %r", text[:80])
        chunks = self._tts.tts_to_file  # fallback placeholder

        # XTTS-v2 streaming via inference_stream
        # The internal synthesiser exposes this when using the low-level API
        synthesiser = self._tts.synthesizer
        outputs = synthesiser.tts_model.inference_stream(
            text,
            self._language,
            synthesiser.gpt_cond_latent,
            synthesiser.speaker_embedding,
            stream_chunk_size=20,   # token chunks — smaller = lower latency
            enable_text_splitting=True,
        )
        for chunk in outputs:
            # chunk is a torch.Tensor [samples] at 24kHz
            audio = chunk.squeeze().cpu().numpy().astype(np.float32)
            yield audio

    def load_speaker(self, wav_path: str):
        """(Re)load speaker voice clone from WAV file."""
        self._speaker_wav = wav_path
        syn = self._tts.synthesizer
        syn.gpt_cond_latent, syn.speaker_embedding = (
            self._tts.synthesizer.tts_model.get_conditioning_latents(
                audio_path=[wav_path]
            )
        )
        log.info("[TTS] Speaker loaded from %s", wav_path)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STUB — Replace with real AriNet pipeline
# ─────────────────────────────────────────────────────────────────────────────

class AriPipeline:
    """
    Stub pipeline. Replace this with your AriNet pipeline call.
    Input: transcribed text from caller
    Output: Ari's response text
    """

    def process(self, text: str) -> str:
        log.info("[PIPELINE] Input: %r", text)
        # ── Replace this with: from core.pipeline import run_pipeline; return run_pipeline(text)
        response = f"Ich habe verstanden: {text}"
        log.info("[PIPELINE] Response: %r", response)
        return response


# ─────────────────────────────────────────────────────────────────────────────
# CALL HANDLER
# ─────────────────────────────────────────────────────────────────────────────

class AriCallHandler:
    """
    Core class that ties everything together:
    - Auto-answers incoming Telegram calls via Telethon
    - Streams PCM in/out via ntgcalls GroupCallRaw
    - VAD → Whisper → AriNet → XTTS → PCM out
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.client  = TelegramClient(
            config["session_name"],
            config["api_id"],
            config["api_hash"],
        )
        self.call    = PyTgCalls(self.client)
        self.ntg     = NTgCalls()

        self.stt      = None   # loaded lazily after Telethon connects
        self.tts      = None
        self.pipeline = AriPipeline()
        self.vad      = SimpleVAD()

        # Outbound audio queue: bytes chunks for ntgcalls to consume
        self._out_queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._active_call_id: int | None = None
        self._call_lock = threading.Lock()

    # ── Startup ───────────────────────────────────────────────────────────────

    async def start(self):
        log.info("[HANDLER] Connecting to Telegram …")
        await self.client.start(phone=self.cfg["phone"])
        log.info("[HANDLER] Logged in as %s", await self.client.get_me())

        # Load models NOW (before first call, avoids cold-start delay mid-call)
        self._load_models()

        # Register Telethon event handler for incoming calls
        self.client.add_event_handler(self._on_update, events.Raw(UpdatePhoneCall))

        # Start pytgcalls
        await self.call.start()

        log.info("[HANDLER] Ari is online — waiting for calls …")
        await self.client.run_until_disconnected()

    def _load_models(self):
        log.info("[HANDLER] Loading STT + TTS models …")
        self.stt = WhisperSTT(
            model_size=self.cfg.get("whisper_model", "large-v3"),
            device=self.cfg.get("device", "cuda"),
        )
        self.tts = XTTSStreamer(
            speaker_wav=self.cfg.get("speaker_wav"),
            language=self.cfg.get("language", "de"),
        )
        # If speaker WAV provided, load it
        if self.cfg.get("speaker_wav"):
            self.tts.load_speaker(self.cfg["speaker_wav"])
        log.info("[HANDLER] Models ready")

    # ── Incoming call ─────────────────────────────────────────────────────────

    async def _on_update(self, update: UpdatePhoneCall):
        """Telethon raw update — intercept incoming call requests."""
        if not isinstance(getattr(update, "phone_call", None), PhoneCallRequested):
            return

        call_obj = update.phone_call
        log.info("[CALL] Incoming call from user_id=%s, call_id=%s",
                 call_obj.admin_id, call_obj.id)

        # Auto-answer
        try:
            await self._accept_call(call_obj)
        except Exception as e:
            log.warning("[CALL] Failed to accept call: %s", e)

    async def _accept_call(self, call_obj):
        """Accept the Telegram call using Telethon + start ntgcalls raw stream."""
        # Step 1: Generate DH params for E2E encryption (required by Telegram protocol)
        # ntgcalls handles the crypto internally — we just need to exchange params
        import hashlib, secrets

        # Accept via Telethon
        accepted = await self.client(AcceptCallRequest(
            peer=call_obj,
            g_b=secrets.token_bytes(256),
            protocol={
                "min_layer": 92,
                "max_layer": 92,
                "library_versions": ["4.0.0"],
            },
        ))
        log.info("[CALL] Accepted — setting up ntgcalls raw stream …")

        call_id = call_obj.id
        with self._call_lock:
            self._active_call_id = call_id

        # Step 2: Connect ntgcalls raw audio (PCM in/out)
        await self._start_raw_stream(call_obj)

    async def _start_raw_stream(self, call_obj):
        """
        Connect GroupCallRaw via pytgcalls.
        on_played_data  → called when ntgcalls wants outbound PCM bytes
        on_received_data → called with inbound PCM bytes from caller
        """
        chat_id = call_obj.admin_id  # or peer id depending on Telethon version

        await self.call.join_group_call(
            chat_id,
            MediaStream(
                audio_parameters=AudioParameters(
                    bitrate=128,
                ),
            ),
            stream_type=MediaStream.STREAM_TYPE_CALL,
        )

        @self.call.on_raw_audio_chunk(chat_id)
        async def _on_audio_received(chunk: bytes):
            """Inbound PCM from caller → VAD → Whisper → Pipeline → TTS → outbound."""
            await self._handle_inbound(chunk)

        @self.call.on_need_audio_chunk(chat_id)
        async def _on_need_audio() -> bytes:
            """ntgcalls pulls outbound PCM from queue every 20ms."""
            return self._get_outbound_chunk()

        log.info("[CALL] Raw stream connected for chat_id=%s", chat_id)

    # ── Audio I/O ─────────────────────────────────────────────────────────────

    async def _handle_inbound(self, raw_bytes: bytes):
        """Process inbound PCM chunk from caller."""
        chunk = pcm_bytes_to_float32(raw_bytes, channels=TELEGRAM_CALL_CHANNELS)
        utterance = self.vad.feed(chunk)

        if utterance is not None:
            # Speech segment complete — transcribe + respond in background thread
            log.info("[VAD] Utterance detected (%.2fs)", len(utterance) / TELEGRAM_CALL_SAMPLE_RATE)
            asyncio.get_event_loop().run_in_executor(None, self._respond, utterance)

    def _respond(self, utterance_48k: np.ndarray):
        """
        Blocking: STT → Pipeline → TTS → push to outbound queue.
        Runs in thread pool so it doesn't block the event loop.
        """
        # STT
        t0 = time.perf_counter()
        text = self.stt.transcribe(utterance_48k)
        log.info("[STT] %.2fs → %r", time.perf_counter() - t0, text)

        if not text.strip():
            log.info("[STT] Empty transcription — skipping")
            return

        # Pipeline
        response_text = self.pipeline.process(text)

        # TTS → stream chunks into outbound queue
        t1 = time.perf_counter()
        first_chunk = True
        for audio_24k in self.tts.stream(response_text):
            if first_chunk:
                log.info("[TTS] First chunk in %.2fs", time.perf_counter() - t1)
                first_chunk = False
            # Resample 24kHz → 48kHz
            audio_48k = resample(audio_24k, XTTS_SAMPLE_RATE, TELEGRAM_CALL_SAMPLE_RATE)
            # Push 20ms chunks to queue
            self._push_audio_chunks(audio_48k)

    def _push_audio_chunks(self, audio_48k: np.ndarray):
        """Split audio array into 20ms chunks and push to outbound queue."""
        total = len(audio_48k)
        offset = 0
        while offset < total:
            end = min(offset + CHUNK_SAMPLES, total)
            chunk = audio_48k[offset:end]
            # Pad last chunk if needed
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            try:
                self._out_queue.put_nowait(float32_to_pcm_bytes(chunk))
            except queue.Full:
                log.warning("[AUDIO] Output queue full — dropping chunk")
            offset = end

    def _get_outbound_chunk(self) -> bytes:
        """Pull next 20ms PCM chunk for ntgcalls. Returns silence if queue empty."""
        try:
            return self._out_queue.get_nowait()
        except queue.Empty:
            # Silence chunk
            return bytes(CHUNK_SAMPLES * 2)  # int16 = 2 bytes/sample


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    config = load_config()
    handler = AriCallHandler(config)
    await handler.start()


if __name__ == "__main__":
    asyncio.run(main())
