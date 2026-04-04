"""
audio_loop.py — ntgcalls 2.1.0 Raw PCM Audio Loop
===================================================
Verified from ntgcalls-2.1.0 C++ source:
  ntgcalls/src/bindings/pythonapi.cpp  — Python API bindings
  ntgcalls/src/media/audio_sink.cpp    — frame size calculation
  ntgcalls/src/media/audio_receiver.cpp — inbound resampling + delivery
  ntgcalls/src/stream_manager.cpp      — set_stream_sources, on_frames flow

KEY FACTS (from source, not guessed):

  AudioSink.frameTime()  = 10ms  (NOT 20ms — common misconception)
  AudioSink.frameSize()  = sampleRate * 16 / 8 / 100 * channelCount
    @ 48000 Hz, 1 ch:    = 48000 * 2 / 100 = 960 bytes = 480 int16 samples

  on_frames callback signature (pythonapi.cpp line 45):
    onFrames(std::function<void(int64_t, StreamManager::Mode, StreamManager::Device, vector<Frame>&)>)
    Python: callback(user_id: int, mode: StreamMode, device: StreamDevice, frames: list[Frame])

  Frame (from pythonapi.cpp):
    frame.data       → bytes  (int16 LE PCM, already resampled to your AudioDescription sample_rate)
    frame.ssrc       → int    (SSRC of the remote sender)
    frame.frame_data → FrameData (unused for audio, width/height = 0)

  AudioReceiver resamples incoming audio TO the sample_rate you specify in AudioDescription.
  If you set AudioDescription(sample_rate=48000), you get 48kHz int16 LE regardless of
  what sample rate the remote client sends.

  set_stream_sources MUST be called BEFORE connect_p2p:
    If called after connect_p2p, streams may not be registered in time.
    Call order: create_p2p_call → init_exchange → set_stream_sources → connect_p2p

  CAPTURE  mode = what WE send (outbound, microphone direction)
  PLAYBACK mode = what WE receive (inbound, speaker direction = caller's voice)

  send_external_frame only works for CAPTURE (outbound).
  on_frames only fires for PLAYBACK (inbound) when MediaSource.EXTERNAL is set on speaker.

  send_external_frame:
    StreamManager::sendExternalFrame checks: externalReaders.contains(device)
    externalReaders is populated by handleCaptureConfig when isExternal=true
    → You must have called set_stream_sources(CAPTURE, MediaDescription(microphone=ext_audio))
    → Only StreamDevice.MICROPHONE is valid for audio outbound

  Frame timing for send_external_frame:
    AudioSink delivers at 10ms intervals (frameRate=100).
    You MUST push frames at 10ms intervals for smooth audio.
    WebRTC jitter buffer handles minor timing variance.
    Use time.perf_counter() not time.sleep(0.01) for precision.

Author: Kevin Kachramanow / NovaMind Studios
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from math import gcd

import numpy as np
import ntgcalls
from scipy.signal import resample_poly


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (derived from AudioSink source)
# ─────────────────────────────────────────────────────────────────────────────

CALL_SR       = 48_000   # Hz — what ntgcalls delivers/expects
CALL_CHANNELS = 1        # mono
CALL_BITS     = 16       # int16 LE — always, non-negotiable

# AudioSink.frameTime() = 10ms → 480 samples per frame at 48kHz mono
FRAME_DURATION_S = 0.010               # 10ms — the ACTUAL ntgcalls frame interval
FRAME_SAMPLES    = CALL_SR // 100      # 480 samples
FRAME_BYTES      = FRAME_SAMPLES * 2   # 960 bytes (int16 = 2 bytes)

WHISPER_SR    = 16_000   # Whisper expects 16kHz
XTTS_SR       = 24_000   # XTTS-v2 native output

# VAD — simple RMS energy gate
VAD_SILENCE_DB   = -38.0  # dB — below this = silence
VAD_SILENCE_SEC  = 0.7    # pause → utterance complete
VAD_MIN_SPEECH_S = 0.25   # ignore very short sounds

# Output queue — max frames buffered (500 × 10ms = 5 seconds)
OUT_QUEUE_MAX = 500


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def pcm_to_f32(raw: bytes) -> np.ndarray:
    """int16 LE PCM bytes → float32 numpy array [-1.0, 1.0]"""
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def f32_to_pcm(audio: np.ndarray) -> bytes:
    """float32 numpy array [-1.0, 1.0] → int16 LE PCM bytes"""
    clamped = np.clip(audio, -1.0, 1.0)
    return (clamped * 32767.0).astype("<i2").tobytes()


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """High-quality polyphase resampling. Returns float32."""
    if src_sr == dst_sr:
        return audio
    g  = gcd(src_sr, dst_sr)
    up = dst_sr // g
    dn = src_sr // g
    return resample_poly(audio, up, dn).astype(np.float32)


def rms_db(audio: np.ndarray) -> float:
    """RMS energy in dB. Returns -100.0 for silence."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return 20.0 * np.log10(max(rms, 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# 1.  configure_stream_sources()
#
#     Must be called BEFORE connect_p2p().
#
#     AudioDescription(media_source, sample_rate, channel_count, input, keep_open=False)
#       media_source:  ntgcalls.MediaSource.EXTERNAL
#       sample_rate:   uint32 — we want 48000
#       channel_count: uint8  — 1 (mono)
#       input:         str    — empty string for EXTERNAL
#       keep_open:     bool   — False (default)
#
#     set_stream_sources(chat_id, direction: StreamMode, media: MediaDescription)
#       CAPTURE  = outbound (what WE send = microphone)
#       PLAYBACK = inbound  (what WE receive = speaker = caller's voice)
#
#     Two separate calls required — one per direction.
# ─────────────────────────────────────────────────────────────────────────────

def configure_stream_sources(ntg: ntgcalls.NTgCalls, user_id: int) -> None:
    """
    Configure raw PCM bidirectional audio for a P2P call.
    Call this AFTER create_p2p_call() / init_exchange() / exchange_keys()
    but BEFORE connect_p2p().

    After this call:
      - send_external_frame(user_id, StreamDevice.MICROPHONE, ...) sends audio OUT
      - on_frames fires with inbound audio from the caller
    """
    ext_audio = ntgcalls.AudioDescription(
        media_source  = ntgcalls.MediaSource.EXTERNAL,
        sample_rate   = CALL_SR,       # 48000
        channel_count = CALL_CHANNELS, # 1
        input         = "",            # empty for EXTERNAL
        keep_open     = False,         # False = clear buffer on reconfigure
    )

    # CAPTURE: outbound — frames we push via send_external_frame()
    ntg.set_stream_sources(
        user_id,
        ntgcalls.StreamMode.CAPTURE,
        ntgcalls.MediaDescription(
            microphone = ext_audio,    # only audio argument needed
            speaker    = None,
            camera     = None,
            screen     = None,
        ),
    )

    # PLAYBACK: inbound — frames ntgcalls delivers via on_frames()
    # AudioReceiver will resample to 48kHz for us regardless of sender's rate.
    ntg.set_stream_sources(
        user_id,
        ntgcalls.StreamMode.PLAYBACK,
        ntgcalls.MediaDescription(
            microphone = None,
            speaker    = ext_audio,   # PLAYBACK direction uses speaker slot
            camera     = None,
            screen     = None,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  VAD — RMS-based Voice Activity Detection
# ─────────────────────────────────────────────────────────────────────────────

class SimpleVAD:
    """
    Accumulates 10ms PCM frames.
    Returns float32 utterance array (48kHz) when speech ends, else None.

    Works directly on what on_frames delivers — no conversion needed
    before feeding. Call feed(chunk_f32) where chunk_f32 is already float32.
    """

    def __init__(self):
        self._buf:     list[np.ndarray] = []
        self._sil_n:   int  = int(VAD_SILENCE_SEC  * CALL_SR)
        self._min_n:   int  = int(VAD_MIN_SPEECH_S * CALL_SR)
        self._sil_acc: int  = 0
        self._in_speech: bool = False

    def feed(self, chunk_f32: np.ndarray) -> np.ndarray | None:
        """
        Feed one 10ms float32 chunk (480 samples @ 48kHz).
        Returns full utterance array when speech segment ends, else None.
        """
        is_speech = rms_db(chunk_f32) > VAD_SILENCE_DB

        if is_speech:
            self._in_speech = True
            self._sil_acc   = 0
            self._buf.append(chunk_f32)
            return None

        if not self._in_speech:
            return None

        # Silence while in speech — accumulate until threshold
        self._buf.append(chunk_f32)
        self._sil_acc += len(chunk_f32)

        if self._sil_acc >= self._sil_n:
            utterance = np.concatenate(self._buf)
            self._buf       = []
            self._sil_acc   = 0
            self._in_speech = False
            if len(utterance) >= self._min_n:
                return utterance

        return None

    def reset(self):
        self._buf       = []
        self._sil_acc   = 0
        self._in_speech = False


# ─────────────────────────────────────────────────────────────────────────────
# 3.  OutboundLoop — sends PCM to ntgcalls at exact 10ms intervals
#
#     send_external_frame signature (pythonapi.cpp):
#       send_external_frame(chat_id: int,
#                           device: StreamDevice,
#                           frame: bytes,
#                           frame_data: FrameData)
#
#     FrameData(absolute_capture_timestamp_ms: int,
#               rotation: int,
#               width: int,
#               height: int)
#       For audio: rotation=0, width=0, height=0
#       absolute_capture_timestamp_ms: monotonic ms timestamp
#
#     frame: bytes — int16 LE PCM
#       Size MUST be exactly FRAME_BYTES (960 bytes = 480 int16 samples @ 48kHz mono)
#       WebRTC's audio pipeline expects consistent frame sizes.
#       Sending wrong size raises ntgcalls.InvalidParams.
#
#     Timing:
#       ntgcalls does NOT call you — you push at your own pace.
#       AudioStreamer.sendData() feeds directly into WebRTC RTCAudioSource.
#       If you push too fast: WebRTC jitter buffer absorbs it (up to ~1s).
#       If you push too slow: caller hears silence/gaps.
#       Target: one FRAME_BYTES frame every 10ms.
#       Use time.perf_counter() — time.sleep(0.01) drifts on Windows.
# ─────────────────────────────────────────────────────────────────────────────

class OutboundLoop:
    """
    Pulls int16 LE PCM frames from a queue and pushes them to ntgcalls
    at exact 10ms intervals using time.perf_counter() for drift compensation.

    Usage:
        loop = OutboundLoop(ntg, user_id)
        loop.start()
        loop.push_audio(audio_48k_f32)   # enqueue XTTS output
        loop.stop()
    """

    _SILENCE = bytes(FRAME_BYTES)   # pre-allocated silence frame

    def __init__(self, ntg: ntgcalls.NTgCalls, user_id: int):
        self._ntg    = ntg
        self._uid    = user_id
        self._q:    queue.Queue[bytes] = queue.Queue(maxsize=OUT_QUEUE_MAX)
        self._stop  = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"out-{user_id}",
        )

    def start(self):  self._thread.start()
    def stop(self):   self._stop.set()

    def push_audio(self, audio_48k: np.ndarray) -> None:
        """
        Enqueue float32 48kHz audio for transmission.
        Splits into FRAME_SAMPLES (480) chunks, pads last frame if needed.
        Call from any thread — queue is thread-safe.
        """
        total, offset = len(audio_48k), 0
        while offset < total:
            end   = min(offset + FRAME_SAMPLES, total)
            chunk = audio_48k[offset:end]
            if len(chunk) < FRAME_SAMPLES:
                # Pad last partial frame with zeros
                chunk = np.concatenate([chunk, np.zeros(FRAME_SAMPLES - len(chunk), dtype=np.float32)])
            try:
                self._q.put_nowait(f32_to_pcm(chunk))
            except queue.Full:
                # Drop oldest frame to make room
                try:
                    self._q.get_nowait()
                    self._q.put_nowait(f32_to_pcm(chunk))
                except queue.Empty:
                    pass
            offset = end

    def _run(self):
        """
        Precise 10ms push loop.
        Uses accumulating next_tick to avoid drift from sleep inaccuracy.
        """
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_tick:
                # Busy-wait the last 1ms for Windows timer resolution
                slack = next_tick - now
                if slack > 0.001:
                    time.sleep(slack - 0.001)
                while time.perf_counter() < next_tick:
                    pass
            next_tick += FRAME_DURATION_S

            # Pull next frame or send silence
            try:
                frame_bytes = self._q.get_nowait()
            except queue.Empty:
                frame_bytes = self._SILENCE

            try:
                # FrameData(absolute_capture_timestamp_ms, rotation, width, height)
                # Audio: rotation=0, width=0, height=0
                fd = ntgcalls.FrameData(
                    int(time.monotonic() * 1000),  # ms timestamp
                    0,   # rotation — irrelevant for audio
                    0,   # width    — irrelevant for audio
                    0,   # height   — irrelevant for audio
                )
                self._ntg.send_external_frame(
                    self._uid,
                    ntgcalls.StreamDevice.MICROPHONE,  # CAPTURE direction
                    frame_bytes,                        # bytes, FRAME_BYTES long
                    fd,
                )
            except ntgcalls.InvalidParams as e:
                # set_stream_sources(CAPTURE) wasn't called, or wrong frame size
                print(f"[OUT] InvalidParams: {e}")
            except ntgcalls.ConnectionNotFound:
                break   # call ended
            except Exception as e:
                print(f"[OUT] send_external_frame error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  InboundRouter — routes on_frames callbacks to the right session
#
#     on_frames callback (from C++ source, confirmed):
#       void(int64_t chat_id,
#            StreamManager::Mode mode,
#            StreamManager::Device device,
#            const vector<Frame>& frames)
#     Python:
#       callback(user_id: int,
#                mode: ntgcalls.StreamMode,
#                device: ntgcalls.StreamDevice,
#                frames: list[ntgcalls.Frame])
#
#     Frame attributes:
#       .data      → bytes  — int16 LE PCM at your configured sample_rate (48000 Hz)
#                             AudioReceiver already resampled it for you.
#                             Each delivery is 480 samples = 960 bytes (10ms @ 48kHz mono)
#       .ssrc      → int    — SSRC of the sender (use 0 for 1-on-1 calls)
#       .frame_data → FrameData — timestamp etc, usually not needed for audio
#
#     IMPORTANT: on_frames fires on ntgcalls internal WebRTC thread.
#     The callback MUST return quickly — no blocking, no heavy work.
#     Hand off to another thread immediately.
# ─────────────────────────────────────────────────────────────────────────────

class AudioSession:
    """
    Manages audio for one active P2P call.

    Call configure_stream_sources() before connect_p2p().
    Call start() after connect_p2p() succeeds.

    on_utterance_ready(audio_48k_f32: np.ndarray) is called in a worker thread
    with the full utterance ready for Whisper. Replace with your STT pipeline.

    push_response_audio(audio_f32: np.ndarray) feeds TTS output back into the call.
    Call this from any thread with XTTS output at any sample rate —
    resampling to 48kHz happens internally.
    """

    def __init__(self,
                 ntg:      ntgcalls.NTgCalls,
                 user_id:  int,
                 on_utterance_ready,  # callable(np.ndarray) — your STT hook
                 src_tts_sr: int = XTTS_SR):
        self._ntg         = ntg
        self._uid         = user_id
        self._on_utt      = on_utterance_ready
        self._tts_sr      = src_tts_sr

        self._vad         = SimpleVAD()
        self._out         = OutboundLoop(ntg, user_id)
        self._worker      = None
        self._utt_q:      queue.Queue[np.ndarray] = queue.Queue()

    def start(self):
        """Start outbound loop + utterance worker thread."""
        self._out.start()
        self._worker = threading.Thread(
            target=self._utterance_worker,
            daemon=True,
            name=f"utt-{self._uid}",
        )
        self._worker.start()

    def stop(self):
        self._out.stop()
        self._utt_q.put(None)   # sentinel to unblock worker

    # ── Called from ntgcalls on_frames — MUST be non-blocking ────────────────

    def on_frames_callback(self,
                           user_id: int,
                           mode: ntgcalls.StreamMode,
                           device: ntgcalls.StreamDevice,
                           frames: list) -> None:
        """
        Register this as:  ntg.on_frames(session.on_frames_callback)

        mode   = StreamMode.PLAYBACK  for inbound audio (caller's voice)
        device = StreamDevice.SPEAKER for inbound audio

        Filters to PLAYBACK only, converts to float32, runs VAD.
        Non-blocking — heavy work goes to _utt_q.
        """
        if mode != ntgcalls.StreamMode.PLAYBACK:
            return
        if user_id != self._uid:
            return

        for frame in frames:
            # frame.data is bytes — int16 LE PCM, 960 bytes = 480 samples @ 48kHz
            raw   = bytes(frame.data)
            chunk = pcm_to_f32(raw)
            utt   = self._vad.feed(chunk)
            if utt is not None:
                try:
                    self._utt_q.put_nowait(utt)
                except queue.Full:
                    pass   # drop if STT can't keep up

    # ── Worker thread: Whisper slot ───────────────────────────────────────────

    def _utterance_worker(self):
        """Dequeues utterances and calls on_utterance_ready. Runs in own thread."""
        while True:
            utt = self._utt_q.get()
            if utt is None:
                break
            try:
                self._on_utt(utt)   # hand to your STT + pipeline + TTS
            except Exception as e:
                print(f"[UTT] utterance handler error: {e}")

    # ── Called after TTS generates audio — push it into the call ─────────────

    def push_response_audio(self, audio_f32: np.ndarray,
                            src_sr: int | None = None) -> None:
        """
        Push TTS output (any sample rate) into the outbound call.
        Resamples to 48kHz internally.

        audio_f32: float32 numpy array
        src_sr:    source sample rate (default: XTTS_SR = 24000)
        """
        sr = src_sr or self._tts_sr
        if sr != CALL_SR:
            audio_f32 = resample(audio_f32, sr, CALL_SR)
        self._out.push_audio(audio_f32)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMPLETE WIRING EXAMPLE
#     Shows exactly how to wire ntgcalls + AudioSession together.
#     Replace the stubs with your Whisper + AriNet pipeline.
# ─────────────────────────────────────────────────────────────────────────────

def _example_full_call_wiring():
    """
    Complete example: P2P call already connected via connect_p2p().
    user_id: Telegram user ID of the peer (used as chat_id in ntgcalls P2P calls).

    CALL FLOW:
    ──────────
    [1] configure_stream_sources(ntg, user_id)    ← BEFORE connect_p2p()
    [2] ntg.connect_p2p(...)
    [3] session = AudioSession(ntg, user_id, on_utterance_ready=my_stt_hook)
    [4] ntg.on_frames(session.on_frames_callback)  ← register ONCE globally
    [5] session.start()

    Then:
    [6] Caller speaks → on_frames fires every 10ms → VAD accumulates
    [7] Silence detected → utterance in queue → my_stt_hook(utt_48k_f32)
    [8] my_stt_hook calls Whisper + pipeline + XTTS, then calls
        session.push_response_audio(tts_output_24k_f32)
    [9] OutboundLoop resamples + sends in 10ms frames
    """

    import ntgcalls

    ntg     = ntgcalls.NTgCalls()
    user_id = 123456789   # replace with actual Telegram user_id

    # ── Step 1: Configure stream sources BEFORE connect_p2p ──────────────────
    configure_stream_sources(ntg, user_id)

    # ── Step 2: connect_p2p (shown as placeholder — already done in ari_voice_call.py)
    # ntg.connect_p2p(user_id, rtc_servers, NTGCALLS_LIBRARY_VERSIONS, p2p_allowed)

    # ── Step 3: Define your STT + pipeline + TTS hook ────────────────────────
    def my_stt_pipeline(utt_48k_f32: np.ndarray):
        """
        Called in a worker thread with a complete utterance.
        utt_48k_f32: float32 numpy array at 48kHz.

        Do:  resample → 16kHz → Whisper → AriNet → XTTS → push back
        """
        # Resample for Whisper
        audio_16k = resample(utt_48k_f32, CALL_SR, WHISPER_SR)
        # audio_16k is float32, numpy array — feed directly to Whisper

        # --- Your Whisper call here ---
        # result = whisper_model.transcribe(audio_16k, language="de")
        # text = result["text"]

        # --- Your AriNet pipeline here ---
        # response_text = pipeline.process(text)

        # --- Your XTTS call here ---
        # for chunk_24k in xtts.inference_stream(response_text, ...):
        #     session.push_response_audio(chunk_24k, src_sr=XTTS_SR)
        pass

    # ── Step 4: Create session ────────────────────────────────────────────────
    session = AudioSession(
        ntg             = ntg,
        user_id         = user_id,
        on_utterance_ready = my_stt_pipeline,
        src_tts_sr      = XTTS_SR,   # 24000
    )

    # ── Step 5: Register on_frames ONCE (global, not per-call) ───────────────
    # The callback receives ALL calls — filter by user_id inside.
    # Must be registered AFTER set_stream_sources, AFTER connect_p2p.
    @ntg.on_frames
    def _on_frames(uid: int, mode: ntgcalls.StreamMode,
                   device: ntgcalls.StreamDevice, frames: list):
        session.on_frames_callback(uid, mode, device, frames)

    # ── Step 6: Start audio loops ─────────────────────────────────────────────
    session.start()

    # Session is now live:
    #   Caller speaks → VAD → Whisper → AriNet → XTTS → back to caller


# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLING REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def resample_inbound_for_whisper(audio_48k_f32: np.ndarray) -> np.ndarray:
    """
    Convert inbound audio (48kHz float32) to Whisper input (16kHz float32).
    gcd(48000, 16000) = 16000 → up=1, down=3 → fast integer resampling.
    """
    return resample(audio_48k_f32, CALL_SR, WHISPER_SR)   # 48k → 16k


def resample_xtts_for_outbound(audio_24k_f32: np.ndarray) -> np.ndarray:
    """
    Convert XTTS output (24kHz float32) to ntgcalls input (48kHz float32).
    gcd(24000, 48000) = 24000 → up=2, down=1 → simple 2× upsample.
    """
    return resample(audio_24k_f32, XTTS_SR, CALL_SR)      # 24k → 48k
