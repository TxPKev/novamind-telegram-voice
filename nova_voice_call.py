"""
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
  Inbound:  ntgcalls on_frame() callback -> 48kHz int16 PCM bytes
            -> float32 -> VAD -> Whisper (16kHz) -> pipeline -> response text
  Outbound: XTTS inference_stream() -> 24kHz float32
            -> resample 48kHz -> int16 PCM bytes
            -> ntgcalls send_external_frame() every 20ms

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
  -> XTTS streams response audio back into call

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
    send_external_frame() - push outbound PCM frame (20ms / 960 samples @ 48kHz)
    on_frames()           - register callback for inbound PCM frames
    send_signaling_data() - relay WebRTC signaling data to remote peer
    on_signal()           - register callback for inbound signaling data

Frame format (matches Telegram/WebRTC standard):
  PCM 16-bit signed little-endian, 48000 Hz, mono
  Frame size: 960 samples = 20ms

License: AGPL-3.0
"""

import asyncio
import logging
import os
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

# ntgcalls / Telegram WebRTC standard
CALL_SAMPLE_RATE   = 48_000    # Hz
CALL_CHANNELS      = 1         # mono
CALL_FRAME_SAMPLES = 960       # 20ms @ 48kHz
CALL_FRAME_BYTES   = CALL_FRAME_SAMPLES * 2  # int16 = 2 bytes/sample

WHISPER_SAMPLE_RATE = 16_000   # Whisper expects 16kHz
XTTS_SAMPLE_RATE    = 24_000   # XTTS-v2 native output

# Voice activity detection
VAD_SILENCE_DB      = -40.0    # dB - below this level = silence
VAD_SILENCE_SECONDS = 0.8      # pause duration that triggers transcription
VAD_MIN_SPEECH_SEC  = 0.3      # ignore utterances shorter than this

# Telegram DH / call protocol
CALL_MIN_LAYER = 65
CALL_MAX_LAYER = 92
CALL_LIBRARY_VERSIONS = ["7.0.0"]   # must be supported by ntgcalls


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
# DH KEY EXCHANGE HELPERS
# -----------------------------------------------------------------------------

def _int_to_bytes_big(n: int, length: int) -> bytes:
    """Big-endian integer -> bytes, zero-padded to length."""
    return n.to_bytes(length, byteorder="big")


def _bytes_to_int(b: bytes) -> int:
    return int.from_bytes(b, byteorder="big")


def _mod_exp(base: int, exp: int, mod: int) -> int:
    return pow(base, exp, mod)


def compute_g_b(dh_config) -> tuple[int, int]:
    """
    Compute g_b = g^b mod p for Telegram DH handshake.
    Returns (b, g_b) where b is our secret exponent.
    dh_config: result of GetDhConfigRequest (has .g, .p, .random)
    """
    p = _bytes_to_int(dh_config.p)
    g = dh_config.g
    # Generate random 256-byte secret exponent
    b = int.from_bytes(os.urandom(256), byteorder="big") % (p - 1)
    g_b = _mod_exp(g, b, p)
    return b, g_b


def compute_shared_key(g_a_or_b: bytes, b: int, dh_config) -> bytes:
    """
    Compute shared key = g_a^b mod p.
    Returns 256-byte key (big-endian, zero-padded).
    """
    p = _bytes_to_int(dh_config.p)
    g_a = _bytes_to_int(g_a_or_b)
    key_int = _mod_exp(g_a, b, p)
    return _int_to_bytes_big(key_int, 256)


# -----------------------------------------------------------------------------
# AUDIO UTILITIES
# -----------------------------------------------------------------------------

def pcm_bytes_to_float32(raw: bytes) -> np.ndarray:
    """int16 PCM bytes -> float32 numpy [-1, 1]."""
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """float32 numpy [-1, 1] -> int16 PCM bytes."""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


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
    XTTS-v2 via inference_stream().
    First audio chunk in ~200ms on RTX 3070.
    Output: 24kHz float32 chunks.
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
        Generator yielding float32 numpy chunks at 24kHz.
        Uses XTTS inference_stream() for low-latency first chunk.
        """
        log.info("[TTS] Synthesising: %r", text[:80])
        syn = self._tts.synthesizer
        for chunk in syn.tts_model.inference_stream(
            text,
            self._lang,
            syn.gpt_cond_latent,
            syn.speaker_embedding,
            stream_chunk_size=20,
            enable_text_splitting=True,
        ):
            yield chunk.squeeze().cpu().numpy().astype(np.float32)


# -----------------------------------------------------------------------------
# PIPELINE STUB
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


# -----------------------------------------------------------------------------
# OUTBOUND AUDIO LOOP
# -----------------------------------------------------------------------------

class OutboundAudioLoop:
    """
    Runs in a background thread.
    Pulls 20ms frames from the queue and pushes them via
    ntgcalls send_external_frame() at the correct 20ms pacing.
    """

    FRAME_INTERVAL = CALL_FRAME_SAMPLES / CALL_SAMPLE_RATE  # 0.020s

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

    def push_audio(self, audio_48k: np.ndarray):
        """Enqueue float32 audio (48kHz) as 20ms PCM frames."""
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
                time.sleep(next_tick - now)
            next_tick += self.FRAME_INTERVAL

            try:
                frame_bytes = self._queue.get_nowait()
            except queue.Empty:
                frame_bytes = self._silence_frame()

            try:
                # ntgcalls native API: send_external_frame(chat_id, device, data, frame_info)
                # device = StreamDevice.MICROPHONE (outbound = microphone stream)
                self._ntg.send_external_frame(
                    self._chat_id,
                    ntgcalls.StreamDevice.MICROPHONE,
                    frame_bytes,
                    ntgcalls.FrameData(
                        absolute_capture_timestamp_ms=int(time.monotonic() * 1000)
                    ),
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
        self._ntg       = ntg
        self._stt       = stt
        self._tts       = tts
        self._pipeline  = pipeline
        self._executor  = executor

        # DH state (set during accept)
        self._b_secret: int | None = None
        self._dh_config             = None

        # Audio
        self._vad     = SimpleVAD()
        self._out_loop: OutboundAudioLoop | None = None

    def store_dh(self, b_secret: int, dh_config):
        self._b_secret  = b_secret
        self._dh_config = dh_config

    def start_audio(self, loop: OutboundAudioLoop):
        self._out_loop = loop
        loop.start()

    def stop(self):
        if self._out_loop:
            self._out_loop.stop()
        try:
            self._ntg.stop(self.peer_id)
        except Exception as e:
            log.warning("[SESSION] stop error: %s", e)

    def on_inbound_frame(self, frame):
        """Called by ntgcalls on_frame() callback - runs in ntgcalls audio thread."""
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

            t1    = time.perf_counter()
            first = True
            for chunk_24k in self._tts.stream(response):
                if first:
                    log.info("[TTS] First chunk in %.2fs", time.perf_counter() - t1)
                    first = False
                chunk_48k = _resample(chunk_24k, XTTS_SAMPLE_RATE, CALL_SAMPLE_RATE)
                if self._out_loop:
                    self._out_loop.push_audio(chunk_48k)
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
        self._pipeline = EchoPipeline()

        # Active sessions keyed by call_id
        self._sessions: dict[int, CallSession] = {}

        import concurrent.futures
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    # Startup

    async def start(self):
        log.info("[HANDLER] Connecting to Telegram ...")
        await self.client.start(phone=self.cfg["phone"])
        me = await self.client.get_me()
        log.info("[HANDLER] Logged in as %s (%s)", me.first_name, me.phone)

        self._load_models()
        self._register_ntgcalls_callbacks()

        self.client.add_event_handler(self._on_update, events.Raw(UpdatePhoneCall))
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

        @self._ntg.on_frame()
        def _on_frame(chat_id: int, frame):
            session = self._sessions.get(chat_id)
            if session:
                session.on_inbound_frame(frame)

        @self._ntg.on_signal()
        def _on_signal(chat_id: int, data: bytes):
            # Relay signaling data back to remote peer via Telethon
            session = self._sessions.get(chat_id)
            if session:
                asyncio.run_coroutine_threadsafe(
                    self._relay_signal(session, data),
                    self.client.loop,
                )

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
            session = self._sessions.pop(call.id, None)
            if session:
                session.stop()

    async def _accept(self, call_req: PhoneCallRequested):
        """DH exchange + AcceptCallRequest."""
        try:
            # Step 1: Get DH config from Telegram
            dh_config = await self.client(GetDhConfigRequest(0, 256))
            b, g_b    = compute_g_b(dh_config)

            # Step 2: Create session, store DH state
            session = CallSession(
                call_req, self._ntg,
                self._stt, self._tts, self._pipeline, self._executor,
            )
            session.store_dh(b, dh_config)
            self._sessions[call_req.id] = session

            # Step 3: AcceptCallRequest
            result = await self.client(AcceptCallRequest(
                peer=InputPhoneCall(call_req.id, call_req.access_hash),
                g_b=_int_to_bytes_big(g_b, 256),
                protocol=PhoneCallProtocol(
                    min_layer=CALL_MIN_LAYER,
                    max_layer=CALL_MAX_LAYER,
                    udp_p2p=True,
                    udp_reflector=True,
                    library_versions=CALL_LIBRARY_VERSIONS,
                ),
            ))
            log.info("[CALL] AcceptCallRequest sent - waiting for PhoneCall confirmation ...")

        except Exception as e:
            log.warning("[CALL] Failed to accept call_id=%s: %s", call_req.id, e)
            self._sessions.pop(call_req.id, None)

    async def _connect_p2p(self, call: PhoneCall):
        """
        Called when PhoneCall arrives (caller confirmed).
        Finalise DH, hand off to ntgcalls, start audio.
        """
        session = self._sessions.get(call.id)
        if session is None:
            log.warning("[CALL] No session for call_id=%s", call.id)
            return

        try:
            b         = session._b_secret
            dh_config = session._dh_config

            # Step 4: Compute shared key from g_a sent by caller
            key_bytes = compute_shared_key(call.g_a_or_b, b, dh_config)

            # Step 5: ntgcalls create_p2p_call for this peer
            peer_id = session.peer_id
            self._ntg.create_p2p_call(peer_id)

            # Step 6: ntgcalls init_exchange + exchange_keys (DH finalisation)
            self._ntg.init_exchange(peer_id, dh_config, call.g_a_or_b)
            auth_params = self._ntg.exchange_keys(
                peer_id,
                call.g_a_or_b,
                call.key_fingerprint,
            )

            # Step 7: Build RTCServer list from PhoneCall connection endpoints
            rtc_servers = _build_rtc_servers(call)

            # Step 8: configure raw PCM mode
            audio_desc = ntgcalls.AudioDescription(
                media_source=ntgcalls.MediaSource.EXTERNAL,
                input="",
                sample_rate=CALL_SAMPLE_RATE,
                channel_count=CALL_CHANNELS,
            )
            self._ntg.set_stream_sources(
                peer_id,
                ntgcalls.StreamMode.CAPTURE,
                ntgcalls.MediaDescription(
                    microphone=audio_desc,
                    speaker=audio_desc,
                ),
            )

            # Step 9: connect_p2p - hands control to ntgcalls WebRTC engine
            self._ntg.connect_p2p(
                peer_id,
                rtc_servers,
                CALL_LIBRARY_VERSIONS,
                call.p2p_allowed,
            )

            # Step 10: Start outbound audio loop
            out_loop = OutboundAudioLoop(self._ntg, peer_id)
            session.start_audio(out_loop)

            log.info("[CALL] P2P connected - bidirectional audio active for user_id=%s", peer_id)

        except Exception as e:
            log.warning("[CALL] P2P connect failed for call_id=%s: %s", call.id, e)
            session = self._sessions.pop(call.id, None)
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
    PhoneCall.connection  - primary PhoneConnection
    PhoneCall.alternative_connections - list of PhoneConnection
    """
    servers = []
    connections = [call.connection] + list(call.alternative_connections or [])
    for conn in connections:
        try:
            srv = ntgcalls.RTCServer(
                id=conn.id,
                ipv4=conn.ip or "",
                ipv6=conn.ipv6 or "",
                port=conn.port,
                username=conn.username or "",
                password=conn.password or "",
                turn=getattr(conn, "turn", True),
                stun=getattr(conn, "stun", False),
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
