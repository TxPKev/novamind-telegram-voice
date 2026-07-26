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

  Stream source ordering (verified against a working implementation):
    CAPTURE  must be set right after create_p2p_call, BEFORE init_exchange.
    PLAYBACK must be set AFTER connect_p2p - only then does on_frames deliver
    inbound frames. Then unmute + resume, or the call connects without audio.
    Call order: create_p2p_call → set CAPTURE → init_exchange → accept
                → exchange_keys → connect_p2p → set PLAYBACK → unmute/resume

  CAPTURE  mode = what WE send (outbound, microphone direction)
  PLAYBACK mode = what WE receive (inbound = caller's voice)

  send_external_frame only works for CAPTURE (outbound).
  on_frames fires for PLAYBACK (inbound) when MediaSource.EXTERNAL is set on the
  microphone slot of the PLAYBACK MediaDescription (inbound frames then arrive
  as device=MICROPHONE, mode=PLAYBACK).

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

Author: Kevin Kachramanow / NovaMind Studio
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


CALL_SR       = 48_000
CALL_CHANNELS = 1
CALL_BITS     = 16

FRAME_DURATION_S = 0.010
FRAME_SAMPLES    = CALL_SR // 100
FRAME_BYTES      = FRAME_SAMPLES * 2

WHISPER_SR    = 16_000
XTTS_SR       = 24_000

VAD_SILENCE_DB   = -38.0
VAD_SILENCE_SEC  = 0.7
VAD_MIN_SPEECH_S = 0.25

OUT_QUEUE_MAX = 500


def pcm_to_f32(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def f32_to_pcm(audio: np.ndarray) -> bytes:
    clamped = np.clip(audio, -1.0, 1.0)
    return (clamped * 32767.0).astype("<i2").tobytes()


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    g  = gcd(src_sr, dst_sr)
    up = dst_sr // g
    dn = src_sr // g
    return resample_poly(audio, up, dn).astype(np.float32)


def rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return 20.0 * np.log10(max(rms, 1e-10))


def _external_audio() -> ntgcalls.AudioDescription:
    return ntgcalls.AudioDescription(
        media_source  = ntgcalls.MediaSource.EXTERNAL,
        sample_rate   = CALL_SR,
        channel_count = CALL_CHANNELS,
        input         = "",
    )


def configure_capture(ntg: ntgcalls.NTgCalls, user_id: int) -> None:
    """Outbound stream source. Call right after create_p2p_call, BEFORE init_exchange."""
    ntg.set_stream_sources(
        user_id,
        ntgcalls.StreamMode.CAPTURE,
        ntgcalls.MediaDescription(microphone=_external_audio()),
    )


def configure_playback(ntg: ntgcalls.NTgCalls, user_id: int) -> None:
    """Inbound stream source. Call AFTER connect_p2p - earlier and on_frames stays silent.
    Note: PLAYBACK also uses the microphone slot; inbound frames arrive as
    device=MICROPHONE, mode=PLAYBACK."""
    ntg.set_stream_sources(
        user_id,
        ntgcalls.StreamMode.PLAYBACK,
        ntgcalls.MediaDescription(microphone=_external_audio()),
    )


class SimpleVAD:

    def __init__(self):
        self._buf:     list[np.ndarray] = []
        self._sil_n:   int  = int(VAD_SILENCE_SEC  * CALL_SR)
        self._min_n:   int  = int(VAD_MIN_SPEECH_S * CALL_SR)
        self._sil_acc: int  = 0
        self._in_speech: bool = False

    def feed(self, chunk_f32: np.ndarray) -> np.ndarray | None:
        is_speech = rms_db(chunk_f32) > VAD_SILENCE_DB
        if is_speech:
            self._in_speech = True
            self._sil_acc   = 0
            self._buf.append(chunk_f32)
            return None
        if not self._in_speech:
            return None
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


class OutboundLoop:

    _SILENCE = bytes(FRAME_BYTES)

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
        total, offset = len(audio_48k), 0
        while offset < total:
            end   = min(offset + FRAME_SAMPLES, total)
            chunk = audio_48k[offset:end]
            if len(chunk) < FRAME_SAMPLES:
                chunk = np.concatenate([chunk, np.zeros(FRAME_SAMPLES - len(chunk), dtype=np.float32)])
            try:
                self._q.put_nowait(f32_to_pcm(chunk))
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self._q.put_nowait(f32_to_pcm(chunk))
                except queue.Empty:
                    pass
            offset = end

    def _run(self):
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            if now < next_tick:
                slack = next_tick - now
                if slack > 0.001:
                    time.sleep(slack - 0.001)
                while time.perf_counter() < next_tick:
                    pass
            next_tick += FRAME_DURATION_S
            try:
                frame_bytes = self._q.get_nowait()
            except queue.Empty:
                frame_bytes = self._SILENCE
            try:
                fd = ntgcalls.FrameData(
                    int(time.monotonic() * 1000),
                    0,
                    0,
                    0,
                )
                self._ntg.send_external_frame(
                    self._uid,
                    ntgcalls.StreamDevice.MICROPHONE,
                    frame_bytes,
                    fd,
                )
            except ntgcalls.InvalidParams as e:
                print(f"[OUT] InvalidParams: {e}")
            except ntgcalls.ConnectionNotFound:
                break
            except Exception as e:
                print(f"[OUT] send_external_frame error: {e}")


class AudioSession:

    def __init__(self,
                 ntg:      ntgcalls.NTgCalls,
                 user_id:  int,
                 on_utterance_ready,
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
        self._out.start()
        self._worker = threading.Thread(
            target=self._utterance_worker,
            daemon=True,
            name=f"utt-{self._uid}",
        )
        self._worker.start()

    def stop(self):
        self._out.stop()
        self._utt_q.put(None)

    def on_frames_callback(self,
                           user_id: int,
                           mode: ntgcalls.StreamMode,
                           device: ntgcalls.StreamDevice,
                           frames: list) -> None:
        if mode != ntgcalls.StreamMode.PLAYBACK:
            return
        if user_id != self._uid:
            return
        for frame in frames:
            raw   = bytes(frame.data)
            chunk = pcm_to_f32(raw)
            utt   = self._vad.feed(chunk)
            if utt is not None:
                try:
                    self._utt_q.put_nowait(utt)
                except queue.Full:
                    pass

    def _utterance_worker(self):
        while True:
            utt = self._utt_q.get()
            if utt is None:
                break
            try:
                self._on_utt(utt)
            except Exception as e:
                print(f"[UTT] utterance handler error: {e}")

    def push_response_audio(self, audio_f32: np.ndarray,
                            src_sr: int | None = None) -> None:
        sr = src_sr or self._tts_sr
        if sr != CALL_SR:
            audio_f32 = resample(audio_f32, sr, CALL_SR)
        self._out.push_audio(audio_f32)


def _example_full_call_wiring():
    import ntgcalls
    ntg     = ntgcalls.NTgCalls()
    user_id = 123456789
    # CAPTURE here (before init_exchange/accept); PLAYBACK only AFTER
    # connect_p2p, followed by unmute + resume - see module docstring.
    configure_capture(ntg, user_id)

    def my_stt_pipeline(utt_48k_f32: np.ndarray):
        audio_16k = resample(utt_48k_f32, CALL_SR, WHISPER_SR)
        pass

    session = AudioSession(
        ntg             = ntg,
        user_id         = user_id,
        on_utterance_ready = my_stt_pipeline,
        src_tts_sr      = XTTS_SR,
    )

    @ntg.on_frames
    def _on_frames(uid: int, mode: ntgcalls.StreamMode,
                   device: ntgcalls.StreamDevice, frames: list):
        session.on_frames_callback(uid, mode, device, frames)

    session.start()


def resample_inbound_for_whisper(audio_48k_f32: np.ndarray) -> np.ndarray:
    return resample(audio_48k_f32, CALL_SR, WHISPER_SR)


def resample_xtts_for_outbound(audio_24k_f32: np.ndarray) -> np.ndarray:
    return resample(audio_24k_f32, XTTS_SR, CALL_SR)
