"""
nova_voice_call.py — Telegram P2P Voice Call Handler
=====================================================
XTTS-v2 + Whisper Large-v3 + ntgcalls native P2P API
Live bidirectional Telegram private call, 100% offline, no cloud, no API keys.

IMPORTANT — READ BEFORE RUNNING
---------------------------------
This PoC uses ntgcalls v2.1.0's native Python bindings directly.
py-tgcalls (the high-level wrapper) does NOT expose private calls or raw PCM
callbacks for 1-on-1 calls in its stable release. We bypass it entirely.

Audio path:
  Inbound:  ntgcalls on_frames() callback -> 48kHz int16 PCM bytes
            -> float32 -> VAD -> Whisper (16kHz) -> pipeline -> response text
  Outbound: XTTS inference() -> 24kHz float32 (full utterance)
            -> resample 48kHz -> int16 PCM bytes
            -> outbound loop paces ntgcalls send_external_frame() every 10ms

Signaling path (MTProto, via Telethon):
  UpdatePhoneCall / PhoneCallRequested
  -> GetDhConfig -> compute g_b -> AcceptCallRequest
  -> PhoneCall (full call object with connection endpoints)
  -> ntgcalls create_p2p_call() -> init_exchange() -> exchange_keys()
  -> connect_p2p() with RTCServer list from PhoneCall

Architecture:
  Caller dials the configured Telegram number
  -> Telethon (UpdatePhoneCall) handles MTProto signaling
  -> ntgcalls handles WebRTC/SRTP transport (native C++)
  -> VAD + Whisper transcribes inbound audio
  -> Pipeline (or stub) generates response
  -> XTTS synthesises the response; outbound loop paces it into the call

Why ntgcalls and not Telethon alone:
  Telegram voice call audio is WebRTC/SRTP - it is peer-to-peer between clients.
  Telethon only handles the MTProto signaling (accept, DH key exchange, endpoints).
  To actually send and receive audio you need a WebRTC stack.
  ntgcalls IS that WebRTC stack, implemented in C++ with Python pybind11 bindings.

Python API note:
  ntgcalls ships pybind11 bindings. Method names follow snake_case convention
  matching the C++ / Go API:
    create_p2p_call()     - initialise P2P session for this peer
    init_exchange()       - start DH handshake with Telegram's dhConfig
    exchange_keys()       - finalise key exchange, get fingerprint
    connect_p2p()         - connect to TURN/STUN servers from PhoneCall object
    set_stream_sources()  - configure raw PCM mode (MediaSource.EXTERNAL)
                            CAPTURE  = outbound (microphone)
                            PLAYBACK = inbound  (speaker = caller's voice)
                            TWO separate calls required, one per direction.
    send_external_frame() - push outbound PCM frame (10ms / 480 samples @ 48kHz)
    on_frames             - register callback for inbound PCM frames
                            Signature: (uid, mode, device, frames: list)
    send_signaling()      - feed inbound MTProto signaling data into ntgcalls
    on_signaling          - register callback for outbound signaling data

Frame format (matches Telegram/WebRTC standard):
  PCM 16-bit signed little-endian, 48000 Hz, mono
  Frame size: 480 samples = 10ms (NOT 20ms — verified in ntgcalls C++ source)

License: AGPL-3.0
"""

import asyncio
import logging
import struct
import sys
import time
import threading
import queue
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

# Telethon
from telethon import TelegramClient, events
from telethon.tl.types import (
    UpdatePhoneCall,
    UpdatePhoneCallSignalingData,
    PhoneCallRequested,
    PhoneCallAccepted,
    PhoneCall,
    PhoneCallDiscarded,
    PhoneCallProtocol,
    InputPhoneCall,
)
from telethon.tl.functions.phone import (
    AcceptCallRequest,
    ConfirmCallRequest,
    DiscardCallRequest,
)
from telethon.tl.functions.messages import GetDhConfigRequest

# ntgcalls (native P2P API)
# Install: pip install ntgcalls==2.1.0 --no-deps
# WARNING: NEVER run pip install without --no-deps in this environment.
# Any package that depends on torch will silently downgrade it to CPU.
try:
    import ntgcalls
except ImportError:
    print("[FATAL] ntgcalls not installed.")
    print("  Run: pip install ntgcalls==2.1.0 --no-deps")
    print("  Do NOT omit --no-deps - it protects your torch/CUDA installation.")
    sys.exit(1)

# Whisper
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    _FASTER_WHISPER = True
except ImportError:
    _FASTER_WHISPER = False
    try:
        import whisper as openai_whisper
    except ImportError:
        print("[FATAL] Neither faster-whisper nor openai-whisper is installed.")
        sys.exit(1)

# XTTS-v2
try:
    from TTS.api import TTS
except ImportError:
    print("[FATAL] Coqui TTS not installed. Run: pip install TTS --no-deps")
    sys.exit(1)

import torch

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("VoiceCall")


# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

# ntgcalls 2.1.0 / Telegram WebRTC standard
# AudioSink.frameTime() = 10ms — verified in ntgcalls C++ source.
# 20ms is a common misconception that breaks send_external_frame().
CALL_SAMPLE_RATE   = 48_000    # Hz
CALL_CHANNELS      = 1         # mono
CALL_FRAME_SAMPLES = 480       # 10ms @ 48kHz
CALL_FRAME_BYTES   = CALL_FRAME_SAMPLES * 2  # int16 LE = 960 bytes

WHISPER_SAMPLE_RATE = 16_000   # Whisper expects 16kHz
XTTS_SAMPLE_RATE    = 24_000   # XTTS-v2 native output

# Voice activity detection
VAD_SILENCE_DB      = -40.0    # dB - below this level = silence
VAD_SILENCE_SECONDS = 0.8      # pause duration that triggers transcription
VAD_MIN_SPEECH_SEC  = 0.3      # ignore utterances shorter than this

# Echo protection: while we speak, our own voice comes back as PLAYBACK
# frames (Telegram buffer reverb). Without this gate the bot transcribes
# its own answer and replies to itself in an endless loop.
ECHO_CARENCE_MS     = 800      # ignore inbound audio this long after speaking

# Call protocol layers/versions come from ntgcalls.NTgCalls.get_protocol()
# at runtime - do not hardcode them here, they go stale.


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

def load_config() -> dict:
    """Load config from config.json. Copy config.example.json to get started."""
    import json
    path = Path(__file__).parent / "config.json"
    if not path.exists():
        log.error("[CONFIG] config.json not found.")
        log.error("  Copy config.example.json -> config.json and fill in your values.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# AUDIO UTILITIES
# -----------------------------------------------------------------------------

def pcm_bytes_to_float32(raw: bytes) -> np.ndarray:
    """int16 PCM bytes -> float32 numpy [-1, 1]."""
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """float32 numpy [-1, 1] -> int16 PCM bytes."""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _resolve_ntg_future(future, deadline: float = 5.0):
    """
    ntgcalls methods may return a Future instead of a plain value.
    Busy-wait it to completion (run this OFF the asyncio event loop -
    ntgcalls futures need the loop running to make progress).
    """
    if future is None:
        return None
    if not hasattr(future, "done"):
        return future
    limit = time.monotonic() + deadline
    while not future.done():
        if time.monotonic() > limit:
            raise TimeoutError(f"ntgcalls Future timeout ({deadline:.0f}s)")
        time.sleep(0.001)
    return future.result()


def _resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """High-quality polyphase resampling via scipy."""
    g   = gcd(from_rate, to_rate)
    up  = to_rate  // g
    dn  = from_rate // g
    return resample_poly(audio, up, dn).astype(np.float32)


def rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return 20.0 * np.log10(max(rms, 1e-10))


def pad_or_trim_to(audio: np.ndarray, n: int) -> np.ndarray:
    if len(audio) >= n:
        return audio[:n]
    return np.pad(audio, (0, n - len(audio)))


# -----------------------------------------------------------------------------
# VAD - energy-based Voice Activity Detection
# -----------------------------------------------------------------------------

class SimpleVAD:
    """
    Accumulates incoming PCM chunks.
    Returns a complete utterance (np.ndarray) when speech ends,
    None on every other feed() call.
    Replace with silero-vad or webrtcvad for production use.
    """

    def __init__(self, sample_rate: int = CALL_SAMPLE_RATE):
        self._sr          = sample_rate
        self._thresh      = VAD_SILENCE_DB
        self._silence_n   = int(VAD_SILENCE_SECONDS * sample_rate)
        self._min_speech  = int(VAD_MIN_SPEECH_SEC  * sample_rate)
        self._buf: list[np.ndarray] = []
        self._sil_count   = 0
        self._speaking    = False

    def feed(self, chunk: np.ndarray) -> np.ndarray | None:
        level    = rms_db(chunk)
        is_speech = level > self._thresh

        if is_speech:
            self._speaking  = True
            self._sil_count = 0
            self._buf.append(chunk)
        elif self._speaking:
            self._buf.append(chunk)
            self._sil_count += len(chunk)
            if self._sil_count >= self._silence_n:
                utterance       = np.concatenate(self._buf)
                self._buf       = []
                self._sil_count = 0
                self._speaking  = False
                if len(utterance) >= self._min_speech:
                    return utterance
        return None

    def reset(self):
        self._buf       = []
        self._sil_count = 0
        self._speaking  = False


# -----------------------------------------------------------------------------
# STT - Whisper
# -----------------------------------------------------------------------------

class WhisperSTT:
    """
    Whisper Large-v3.
    Loads once at startup. Uses faster-whisper (int8) when available.
    """

    def __init__(self, model_size: str = "large-v3",
                 device: str = "cuda", language: str = "de"):
        self._lang = language
        if _FASTER_WHISPER:
            log.info("[STT] Loading faster-whisper %s (int8_float16) ...", model_size)
            self._model = FasterWhisperModel(
                model_size, device=device, compute_type="int8_float16"
            )
            self._mode = "faster"
        else:
            log.info("[STT] Loading openai-whisper %s ...", model_size)
            self._model = openai_whisper.load_model(model_size, device=device)
            self._mode = "openai"
        log.info("[STT] Ready (%s)", self._mode)

    def transcribe(self, audio_48k: np.ndarray) -> str:
        """48kHz float32 -> transcribed text."""
        audio_16k = _resample(audio_48k, CALL_SAMPLE_RATE, WHISPER_SAMPLE_RATE)
        if self._mode == "faster":
            segs, _ = self._model.transcribe(
                audio_16k, language=self._lang,
                beam_size=5, vad_filter=True,
            )
            return " ".join(s.text for s in segs).strip()
        else:
            result = self._model.transcribe(
                audio_16k, language=self._lang,
                fp16=torch.cuda.is_available(),
            )
            return result["text"].strip()


# -----------------------------------------------------------------------------
# TTS - XTTS-v2
# -----------------------------------------------------------------------------

class XTTSStreamer:
    """
    XTTS-v2 speech synthesis. Uses the non-streaming inference() (see stream()
    for why streaming is avoided). Output: 24kHz float32, one utterance per call;
    the outbound loop paces it into 10 ms frames.
    """

    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(self, speaker_wav: str | None = None, language: str = "de"):
        log.info("[TTS] Loading XTTS-v2 ...")
        self._tts   = TTS(model_name=self.MODEL_NAME).to("cuda")
        self._lang  = language
        if speaker_wav:
            self.load_speaker(speaker_wav)
        log.info("[TTS] Ready")

    def load_speaker(self, wav_path: str):
        syn = self._tts.synthesizer
        (syn.gpt_cond_latent,
         syn.speaker_embedding) = syn.tts_model.get_conditioning_latents(
            audio_path=[wav_path]
        )
        log.info("[TTS] Speaker loaded: %s", wav_path)

    def stream(self, text: str):
        """
        Generator yielding float32 numpy audio at 24kHz.

        NOTE: XTTS `inference_stream()` raises "'int' object has no attribute
        'device'" on TTS 0.22.0 / transformers 4.46.3. We use the non-streaming
        `inference()` instead: it returns the full utterance, which the outbound
        loop paces into 10 ms frames anyway, and it works on both that pairing
        and the current coqui-tts install (see README "Install").
        """
        log.info("[TTS] Synthesising: %r", text[:80])
        syn = self._tts.synthesizer
        out = syn.tts_model.inference(
            text,
            self._lang,
            syn.gpt_cond_latent,
            syn.speaker_embedding,
            temperature=0.7,
            enable_text_splitting=True,
        )
        wav = out["wav"] if isinstance(out, dict) else out
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        if wav.size:
            yield wav


# -----------------------------------------------------------------------------
# PIPELINES - the "brain" is swappable, bring your own offline model
# -----------------------------------------------------------------------------

class EchoPipeline:
    """
    Stub pipeline - replace process() with your own logic.
    Default behaviour: echo the transcribed text back to the caller.
    """

    def process(self, text: str) -> str:
        log.info("[PIPELINE] Input: %r", text)
        response = f"You said: {text}"
        log.info("[PIPELINE] Response: %r", response)
        return response


class OllamaPipeline:
    """
    Optional pipeline: send the transcribed text to a local Ollama instance
    (https://ollama.com) and speak the model's reply back to the caller.

    Activate in config.json:
        "pipeline":     "ollama",
        "ollama_model": "qwen2.5:3b"

    Requires a running Ollama server (default http://localhost:11434).
    Uses only the Python standard library - no extra dependencies.
    """

    DEFAULT_HOST = "http://localhost:11434"

    def __init__(self, config: dict):
        self._host  = config.get("ollama_host", self.DEFAULT_HOST).rstrip("/")
        self._model = config.get("ollama_model", "qwen2.5:3b")
        # Keep replies short - they are spoken back over the call.
        self._system = config.get(
            "ollama_system",
            "You are a voice assistant on a phone call. "
            "Answer briefly, in one or two spoken sentences.",
        )

    def process(self, text: str) -> str:
        import json as _json
        import urllib.request
        import urllib.error
        log.info("[PIPELINE] Input: %r", text)
        payload = _json.dumps({
            "model":  self._model,
            "prompt": text,
            "system": self._system,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self._host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            # Generous timeout: first request loads the model into memory.
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
            response = (data.get("response") or "").strip()
            if not response:
                log.error("[PIPELINE] Ollama returned an empty response")
                response = "I did not get an answer from the model."
        except (urllib.error.URLError, OSError, ValueError) as e:
            log.error("[PIPELINE] Ollama request failed: %s", e)
            response = "Sorry, my language model is not reachable right now."
        log.info("[PIPELINE] Response: %r", response)
        return response


# -----------------------------------------------------------------------------
# OUTBOUND AUDIO LOOP
# -----------------------------------------------------------------------------

class OutboundAudioLoop:
    """
    Runs in a background thread.
    Pulls 10ms frames from the queue and pushes them via
    ntgcalls send_external_frame() at the correct 10ms pacing.
    Drift-corrected via busy-wait on the last 1ms (Windows time.sleep resolution).
    """

    FRAME_INTERVAL = CALL_FRAME_SAMPLES / CALL_SAMPLE_RATE  # 0.010s

    def __init__(self, ntg: "ntgcalls.NTgCalls", chat_id: int):
        self._ntg     = ntg
        self._chat_id = chat_id
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=500)
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def queue_empty(self) -> bool:
        return self._queue.empty()

    def push_audio(self, audio_48k: np.ndarray):
        """Enqueue float32 audio (48kHz) as 10ms PCM frames."""
        total  = len(audio_48k)
        offset = 0
        while offset < total:
            end   = min(offset + CALL_FRAME_SAMPLES, total)
            chunk = audio_48k[offset:end]
            if len(chunk) < CALL_FRAME_SAMPLES:
                chunk = np.pad(chunk, (0, CALL_FRAME_SAMPLES - len(chunk)))
            try:
                self._queue.put_nowait(float32_to_pcm_bytes(chunk))
            except queue.Full:
                log.warning("[OUT] Output queue full - dropping frame")
            offset = end

    def _silence_frame(self) -> bytes:
        return bytes(CALL_FRAME_BYTES)

    def _run(self):
        log.info("[OUT] Audio loop started")
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_tick:
                slack = next_tick - now
                if slack > 0.001:
                    time.sleep(slack - 0.001)
                # busy-wait the last 1ms — Windows time.sleep() resolution
                while time.perf_counter() < next_tick:
                    pass
            next_tick += self.FRAME_INTERVAL

            try:
                frame_bytes = self._queue.get_nowait()
            except queue.Empty:
                frame_bytes = self._silence_frame()

            try:
                # ntgcalls native API: send_external_frame(chat_id, device, data, frame_info)
                # device = StreamDevice.MICROPHONE (outbound = microphone stream)
                # FrameData takes 4 positional args (ts, width, height, rotation) —
                # width/height/rotation are 0 for audio. Same as audio_loop.py.
                self._ntg.send_external_frame(
                    self._chat_id,
                    ntgcalls.StreamDevice.MICROPHONE,
                    frame_bytes,
                    ntgcalls.FrameData(int(time.monotonic() * 1000), 0, 0, 0),
                )
            except Exception as e:
                log.warning("[OUT] send_external_frame error: %s", e)

        log.info("[OUT] Audio loop stopped")


# -----------------------------------------------------------------------------
# CALL SESSION - one per active call
# -----------------------------------------------------------------------------

class CallSession:
    """
    Manages a single active call:
    - DH key exchange state
    - ntgcalls P2P session
    - Inbound VAD + STT
    - Outbound TTS audio loop
    """

    def __init__(self, call_obj: PhoneCallRequested,
                 ntg: "ntgcalls.NTgCalls",
                 stt: WhisperSTT,
                 tts: XTTSStreamer,
                 pipeline: EchoPipeline,
                 executor):
        self.call_id    = call_obj.id
        self.access_hash = call_obj.access_hash
        self.peer_id    = call_obj.admin_id
        # ntgcalls-internal call id - create_p2p_call's return value.
        # NOT necessarily the user id; peer_id is only the fallback.
        self.ntg_call_id = call_obj.admin_id
        self._ntg       = ntg
        self._stt       = stt
        self._tts       = tts
        self._pipeline  = pipeline
        self._executor  = executor

        # Signaling state: data arriving before connect_p2p must be queued
        # (same pattern as the proven implementation - sending it early is lost)
        self.connection_initialized = False
        self.signaling_queue: list[bytes] = []
        # Guard: connect_p2p must run once even though Telegram re-sends PhoneCall
        self.p2p_started = False

        # Echo protection state (see ECHO_CARENCE_MS)
        self._speaking             = False
        self._speaking_released_at = 0.0

        # Audio
        self._vad     = SimpleVAD()
        self._out_loop: OutboundAudioLoop | None = None

    def start_audio(self, loop: OutboundAudioLoop):
        self._out_loop = loop
        loop.start()

    def stop(self):
        if self._out_loop:
            self._out_loop.stop()
        try:
            self._ntg.stop(self.ntg_call_id)
        except Exception as e:
            log.warning("[SESSION] stop error: %s", e)

    def on_inbound_frame(self, frame):
        """Called per frame from the global on_frames callback (PLAYBACK only)."""
        # Echo gates: while we speak (and shortly after), inbound frames are
        # our own voice reverberating back - discard, never transcribe.
        if self._speaking:
            self._vad.reset()
            return
        if (time.monotonic() - self._speaking_released_at) * 1000 < ECHO_CARENCE_MS:
            return
        raw_bytes = bytes(frame.data)
        chunk     = pcm_bytes_to_float32(raw_bytes)
        utterance = self._vad.feed(chunk)
        if utterance is not None:
            dur = len(utterance) / CALL_SAMPLE_RATE
            log.info("[VAD] Utterance detected (%.2fs)", dur)
            # Dispatch to thread pool - do not block audio callback
            self._executor.submit(self._respond, utterance)

    def _respond(self, utterance_48k: np.ndarray):
        """STT -> Pipeline -> TTS -> push to outbound queue (runs in thread pool)."""
        try:
            t0   = time.perf_counter()
            text = self._stt.transcribe(utterance_48k)
            log.info("[STT] %.2fs -> %r", time.perf_counter() - t0, text)
            if not text.strip():
                return

            response = self._pipeline.process(text)

            self._speaking = True
            try:
                t1    = time.perf_counter()
                first = True
                for chunk_24k in self._tts.stream(response):
                    if first:
                        log.info("[TTS] First chunk in %.2fs", time.perf_counter() - t1)
                        first = False
                    chunk_48k = _resample(chunk_24k, XTTS_SAMPLE_RATE, CALL_SAMPLE_RATE)
                    if self._out_loop:
                        self._out_loop.push_audio(chunk_48k)
                # Wait until the queued audio has actually been sent -
                # only then does the echo carence window start.
                deadline = time.monotonic() + 60.0
                while (self._out_loop and not self._out_loop.queue_empty()
                       and time.monotonic() < deadline):
                    time.sleep(0.05)
            finally:
                # Everything that arrived while we spoke is echo - drop it.
                self._vad.reset()
                self._speaking             = False
                self._speaking_released_at = time.monotonic()
        except Exception as e:
            log.warning("[SESSION] _respond error: %s", e)


# -----------------------------------------------------------------------------
# CALL HANDLER - top-level orchestrator
# -----------------------------------------------------------------------------

class CallHandler:
    """
    Orchestrates everything:
    - Telethon for MTProto signaling
    - ntgcalls for WebRTC P2P audio
    - Per-call CallSession instances
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.client = TelegramClient(
            config["session_name"],
            config["api_id"],
            config["api_hash"],
        )
        # ntgcalls native instance - one per process
        self._ntg = ntgcalls.NTgCalls()

        self._stt      = None
        self._tts      = None
        # Swappable brain: "echo" (default) or "ollama" - set "pipeline" in config.json
        if config.get("pipeline", "echo") == "ollama":
            self._pipeline = OllamaPipeline(config)
        else:
            self._pipeline = EchoPipeline()

        # Active sessions keyed by peer_id (= chat_id in ntgcalls P2P)
        self._sessions: dict[int, CallSession] = {}
        # Reverse lookup: call_id -> peer_id (for PhoneCallDiscarded routing)
        self._call_id_to_peer: dict[int, int] = {}
        # ntgcalls-internal call id -> peer_id (they are NOT guaranteed equal;
        # ntgcalls callbacks report the internal id)
        self._ntg_to_peer: dict[int, int] = {}

        import concurrent.futures
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        # The running asyncio loop, captured in start(). ntgcalls callbacks fire
        # on a native thread and must hop back onto THIS loop to talk to Telethon.
        self._loop = None

    async def _ntg_call(self, method, *args, deadline: float = 5.0, **kwargs):
        """Run an ntgcalls method off the event loop and resolve its Future."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: _resolve_ntg_future(method(*args, **kwargs), deadline))

    # Startup

    async def start(self):
        # Capture the running loop so native ntgcalls callbacks can schedule
        # coroutines back onto it (client.loop is unreliable across versions).
        self._loop = asyncio.get_running_loop()
        log.info("[HANDLER] Connecting to Telegram ...")
        await self.client.start(phone=self.cfg["phone"])
        me = await self.client.get_me()
        log.info("[HANDLER] Logged in as %s (%s)", me.first_name, me.phone)

        self._load_models()
        self._register_ntgcalls_callbacks()

        self.client.add_event_handler(self._on_update, events.Raw(UpdatePhoneCall))
        self.client.add_event_handler(
            self._on_signaling_data, events.Raw(UpdatePhoneCallSignalingData))
        log.info("[HANDLER] Voice handler online - waiting for calls ...")
        await self.client.run_until_disconnected()

    def _load_models(self):
        log.info("[HANDLER] Loading STT + TTS models ...")
        self._stt = WhisperSTT(
            model_size=self.cfg.get("whisper_model", "large-v3"),
            device=self.cfg.get("device", "cuda"),
            language=self.cfg.get("language", "de"),
        )
        self._tts = XTTSStreamer(
            speaker_wav=self.cfg.get("speaker_wav"),
            language=self.cfg.get("language", "de"),
        )
        log.info("[HANDLER] Models ready")

    def _register_ntgcalls_callbacks(self):
        """Register ntgcalls-level callbacks that apply to all calls."""

        @self._ntg.on_frames
        def _on_frames(uid: int,
                       mode: ntgcalls.StreamMode,
                       device: ntgcalls.StreamDevice,
                       frames: list):
            # PLAYBACK = inbound (caller's voice). CAPTURE never fires here.
            if mode != ntgcalls.StreamMode.PLAYBACK:
                return
            # uid is the ntgcalls-internal call id - map to peer first
            session = self._sessions.get(self._ntg_to_peer.get(uid, uid))
            if session is None:
                return
            for frame in frames:
                session.on_inbound_frame(frame)

        @self._ntg.on_signaling
        def _on_signaling(chat_id: int, data: bytes):
            # Relay signaling data back to Telegram via Telethon. This fires on a
            # native ntgcalls thread - hop onto the captured asyncio loop.
            session = self._sessions.get(self._ntg_to_peer.get(chat_id, chat_id))
            if session and self._loop:
                asyncio.run_coroutine_threadsafe(
                    self._relay_signal(session, data),
                    self._loop,
                )

        @self._ntg.on_connection_change
        def _on_connection_change(chat_id: int, network_info):
            # State arrives as enum in-process; normalise to the bare name.
            state = getattr(network_info, "state", None)
            state_name = str(state).rsplit(".", 1)[-1] if state is not None else "?"
            log.info("[CALL] Connection state: %s (chat_id=%s)", state_name, chat_id)

    async def _relay_signal(self, session: CallSession, data: bytes):
        """Send WebRTC signaling data back through MTProto."""
        try:
            from telethon.tl.functions.phone import SendSignalingDataRequest
            await self.client(SendSignalingDataRequest(
                peer=InputPhoneCall(session.call_id, session.access_hash),
                data=data,
            ))
        except Exception as e:
            log.warning("[SIGNAL] relay error: %s", e)

    async def _on_signaling_data(self, update: UpdatePhoneCallSignalingData):
        """Inbound signaling from Telegram -> ntgcalls. Queued until connect_p2p ran."""
        peer_id = self._call_id_to_peer.get(update.phone_call_id)
        session = self._sessions.get(peer_id) if peer_id is not None else None
        if session is None:
            log.warning("[SIGNAL] no session for call_id=%s", update.phone_call_id)
            return
        data = bytes(update.data)
        if not session.connection_initialized:
            session.signaling_queue.append(data)
            log.info("[SIGNAL] queued before connect_p2p: %d bytes", len(data))
            return
        try:
            await self._ntg_call(self._ntg.send_signaling, session.ntg_call_id, data)
        except Exception as e:
            log.warning("[SIGNAL] send_signaling error: %s", e)

    # Incoming call

    async def _on_update(self, update: UpdatePhoneCall):
        call = getattr(update, "phone_call", None)
        if call is None:
            return

        if isinstance(call, PhoneCallRequested):
            log.info("[CALL] Incoming from user_id=%s call_id=%s",
                     call.admin_id, call.id)
            await self._accept(call)

        elif isinstance(call, PhoneCall):
            # Full PhoneCall object arrives after remote confirms -> connect ntgcalls
            log.info("[CALL] PhoneCall confirmed: id=%s", call.id)
            await self._connect_p2p(call)

        elif isinstance(call, PhoneCallDiscarded):
            log.info("[CALL] Discarded: id=%s reason=%s", call.id, call.reason)
            peer_id = self._call_id_to_peer.pop(call.id, None)
            if peer_id is not None:
                session = self._sessions.pop(peer_id, None)
                if session:
                    self._ntg_to_peer.pop(session.ntg_call_id, None)
                    session.stop()

    async def _accept(self, call_req: PhoneCallRequested):
        """
        Accept an incoming call. The DH handshake is done entirely by ntgcalls
        (init_exchange returns our g_b) - no Python-side DH math.
        Order matters: create_p2p_call -> CAPTURE stream -> init_exchange -> accept.
        """
        try:
            # Step 1: Create session
            session = CallSession(
                call_req, self._ntg,
                self._stt, self._tts, self._pipeline, self._executor,
            )
            peer_id = session.peer_id
            self._sessions[peer_id] = session
            self._call_id_to_peer[call_req.id] = peer_id

            # Step 2: create_p2p_call MUST come before init_exchange.
            # Return value = ntgcalls-internal call id (NOT the user id!);
            # some versions return None - then the user id is the fallback.
            ntg_id = await self._ntg_call(self._ntg.create_p2p_call, peer_id)
            if not ntg_id:
                ntg_id = peer_id
            session.ntg_call_id = ntg_id
            self._ntg_to_peer[ntg_id] = peer_id

            # Step 3: CAPTURE (outbound) stream source right after create_p2p_call -
            # BEFORE init_exchange and connect_p2p. PLAYBACK is set after connect_p2p.
            await self._ntg_call(
                self._ntg.set_stream_sources,
                ntg_id,
                ntgcalls.StreamMode.CAPTURE,
                ntgcalls.MediaDescription(
                    microphone=ntgcalls.AudioDescription(
                        media_source=ntgcalls.MediaSource.EXTERNAL,
                        sample_rate=CALL_SAMPLE_RATE,
                        channel_count=CALL_CHANNELS,
                        input="",
                    ),
                ),
            )

            # Step 4: DH config from Telegram, handed to ntgcalls
            dh_config = await self.client(GetDhConfigRequest(
                version=0, random_length=256))
            exchange_result = await self._ntg_call(
                self._ntg.init_exchange,
                user_id=ntg_id,
                dh_config=ntgcalls.DhConfig(
                    g=dh_config.g,
                    p=dh_config.p,
                    random=dh_config.random,
                ),
                g_a_hash=call_req.g_a_hash,
            )
            # Result can be AuthParams (has .g_b) or raw bytes
            if hasattr(exchange_result, "g_b"):
                g_b = exchange_result.g_b
            elif isinstance(exchange_result, bytes):
                g_b = exchange_result
            else:
                g_b = bytes(exchange_result)
            log.info("[CALL] init_exchange done, g_b length: %d", len(g_b))

            # Step 5: AcceptCallRequest with protocol straight from ntgcalls
            protocol = ntgcalls.NTgCalls.get_protocol()
            await self.client(AcceptCallRequest(
                peer=InputPhoneCall(call_req.id, call_req.access_hash),
                g_b=g_b,
                protocol=PhoneCallProtocol(
                    min_layer=protocol.min_layer,
                    max_layer=protocol.max_layer,
                    udp_p2p=protocol.udp_p2p,
                    udp_reflector=protocol.udp_reflector,
                    library_versions=list(protocol.library_versions),
                ),
            ))
            log.info("[CALL] AcceptCallRequest sent - waiting for PhoneCall confirmation ...")

        except Exception as e:
            log.warning("[CALL] Failed to accept call_id=%s: %s", call_req.id, e)
            peer_id = self._call_id_to_peer.pop(call_req.id, None)
            if peer_id is not None:
                self._sessions.pop(peer_id, None)

    async def _connect_p2p(self, call: PhoneCall):
        """
        Called when PhoneCall arrives (caller confirmed).
        Finalise DH, hand off to ntgcalls, start audio.
        """
        peer_id = self._call_id_to_peer.get(call.id)
        if peer_id is None:
            log.warning("[CALL] No peer mapping for call_id=%s", call.id)
            return
        session = self._sessions.get(peer_id)
        if session is None:
            log.warning("[CALL] No session for peer_id=%s", peer_id)
            return

        # Telegram re-sends the PhoneCall (confirmed) update several times.
        # connect_p2p must run exactly once - a second run makes ntgcalls raise
        # "Connection already made", and the error path would tear down the
        # already-established call. Guard against re-entry.
        if session.p2p_started:
            log.info("[CALL] Duplicate PhoneCall update for call_id=%s - ignored", call.id)
            return
        session.p2p_started = True

        try:
            ntg_id = session.ntg_call_id

            # Step 6: exchange_keys finalises the DH handshake inside ntgcalls
            # (init_exchange already ran during _accept)
            await self._ntg_call(
                self._ntg.exchange_keys,
                user_id=ntg_id,
                g_a_or_b=call.g_a_or_b,
                fingerprint=call.key_fingerprint,
            )

            # Step 7: Build RTCServer list from PhoneCall connection endpoints
            rtc_servers = _build_rtc_servers(call)
            if not rtc_servers:
                log.warning("[CALL] No RTC servers in PhoneCall object")
                return

            # Step 8: connect_p2p - hands control to ntgcalls WebRTC engine
            protocol = ntgcalls.NTgCalls.get_protocol()
            await self._ntg_call(
                self._ntg.connect_p2p,
                user_id=ntg_id,
                servers=rtc_servers,
                versions=list(protocol.library_versions),
                p2p_allowed=True,
                deadline=12.0,
            )

            # Flush signaling data that arrived before connect_p2p was ready
            session.connection_initialized = True
            for queued in session.signaling_queue:
                try:
                    await self._ntg_call(self._ntg.send_signaling, ntg_id, queued)
                except Exception as e:
                    log.warning("[SIGNAL] queued relay error: %s", e)
            session.signaling_queue.clear()

            # Step 9: PLAYBACK (inbound) stream source AFTER connect_p2p -
            # only then does on_frames deliver inbound frames.
            await self._ntg_call(
                self._ntg.set_stream_sources,
                ntg_id,
                ntgcalls.StreamMode.PLAYBACK,
                ntgcalls.MediaDescription(
                    microphone=ntgcalls.AudioDescription(
                        media_source=ntgcalls.MediaSource.EXTERNAL,
                        sample_rate=CALL_SAMPLE_RATE,
                        channel_count=CALL_CHANNELS,
                        input="",
                    ),
                ),
            )

            # Step 10: unmute + resume - without these the call connects
            # but carries no audio (documented in the proven implementation)
            await self._ntg_call(self._ntg.unmute, ntg_id)
            await self._ntg_call(self._ntg.resume, ntg_id)

            # Step 11: Start outbound audio loop (doubles as silence keepalive -
            # 2-3s without frames would trigger PhoneCallDiscarded)
            out_loop = OutboundAudioLoop(self._ntg, ntg_id)
            session.start_audio(out_loop)

            log.info("[CALL] P2P connected - bidirectional audio active for user_id=%s", peer_id)

        except Exception as e:
            log.warning("[CALL] P2P connect failed for call_id=%s: %s", call.id, e)
            peer_id = self._call_id_to_peer.pop(call.id, None)
            if peer_id is not None:
                session = self._sessions.pop(peer_id, None)
                if session:
                    session.stop()
            # Discard the call cleanly
            try:
                await self.client(DiscardCallRequest(
                    peer=InputPhoneCall(call.id, call.access_hash),
                    duration=0,
                    reason=None,
                    connection_id=0,
                ))
            except Exception:
                pass


def _build_rtc_servers(call: PhoneCall) -> list:
    """
    Convert Telegram PhoneCall connection endpoints to ntgcalls RTCServer objects.
    PhoneCall.connections is the endpoint list (PhoneConnection / PhoneConnectionWebrtc).
    RTCServer requires the tcp argument; peer_tag is needed for reflector relays.
    """
    servers = []
    for conn in (getattr(call, "connections", None) or []):
        try:
            srv = ntgcalls.RTCServer(
                id=conn.id,
                ipv4=conn.ip if hasattr(conn, "ip") else conn.ipv4,
                ipv6=getattr(conn, "ipv6", "") or "",
                port=conn.port,
                username=getattr(conn, "username", None),
                password=getattr(conn, "password", None),
                turn=getattr(conn, "turn", False),
                stun=getattr(conn, "stun", False),
                tcp=getattr(conn, "tcp", False),
                peer_tag=getattr(conn, "peer_tag", None),
            )
            servers.append(srv)
        except Exception as e:
            log.warning("[RTC] Could not build RTCServer from connection: %s", e)
    return servers


# -----------------------------------------------------------------------------
# ENTRYPOINT
# -----------------------------------------------------------------------------

async def main():
    config  = load_config()
    handler = CallHandler(config)
    await handler.start()


if __name__ == "__main__":
    asyncio.run(main())
