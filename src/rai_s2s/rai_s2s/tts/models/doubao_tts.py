# Copyright (C) 2026
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import base64
import io
import os
import uuid
from typing import Any, Tuple

import requests
from pydub import AudioSegment

from rai_s2s.tts.models import TTSModel, TTSModelError


class DoubaoTTS(TTSModel):
    """Volcengine/Doubao HTTP TTS model."""

    def __init__(
        self,
        *,
        app_id: str | None = None,
        token: str | None = None,
        cluster: str | None = None,
        voice_type: str | None = None,
        url: str | None = None,
        encoding: str = "wav",
        audio_rate: int = 24000,
        speed_ratio: float = 1.0,
        volume_ratio: float = 1.0,
        pitch_ratio: float = 1.0,
        timeout: float = 30.0,
    ):
        self.app_id = app_id or os.getenv("DOUBAO_TTS_APP_ID") or os.getenv("DOUBAO_APP_ID")
        self.token = token or os.getenv("DOUBAO_TTS_TOKEN") or os.getenv("DOUBAO_TOKEN")
        self.cluster = cluster or os.getenv("DOUBAO_TTS_CLUSTER") or "volcano_tts"
        self.voice_type = voice_type or os.getenv("DOUBAO_TTS_VOICE_TYPE") or ""
        self.url = url or os.getenv("DOUBAO_TTS_URL") or "https://openspeech.bytedance.com/api/v1/tts"
        self.encoding = encoding
        self.audio_rate = audio_rate
        self.speed_ratio = speed_ratio
        self.volume_ratio = volume_ratio
        self.pitch_ratio = pitch_ratio
        self.timeout = timeout

        if not self.app_id:
            raise TTSModelError("DoubaoTTS requires DOUBAO_TTS_APP_ID or DOUBAO_APP_ID.")
        if not self.token:
            raise TTSModelError("DoubaoTTS requires DOUBAO_TTS_TOKEN or DOUBAO_TOKEN.")
        if not self.voice_type:
            raise TTSModelError("DoubaoTTS requires DOUBAO_TTS_VOICE_TYPE.")

    def get_speech(self, text: str) -> AudioSegment:
        payload = {
            "app": {
                "appid": self.app_id,
                "token": self.token,
                "cluster": self.cluster,
            },
            "user": {"uid": "rai_s2s"},
            "audio": {
                "voice_type": self.voice_type,
                "encoding": self.encoding,
                "rate": self.audio_rate,
                "speed_ratio": self.speed_ratio,
                "volume_ratio": self.volume_ratio,
                "pitch_ratio": self.pitch_ratio,
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "text_type": "plain",
                "operation": "query",
            },
        }
        headers = {
            "Authorization": f"Bearer;{self.token}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                self.url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise TTSModelError(f"Doubao TTS request failed: {exc}") from exc

        audio_bytes = self._extract_audio_bytes(response)
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=self.encoding)
        if self.sample_rate == -1:
            return audio
        return self._resample(audio)

    def get_tts_params(self) -> Tuple[int, int]:
        data = self.get_speech("你好")
        return data.frame_rate, data.channels

    def _extract_audio_bytes(self, response: requests.Response) -> bytes:
        content_type = response.headers.get("Content-Type", "")
        if "audio" in content_type:
            return response.content

        try:
            result: dict[str, Any] = response.json()
        except ValueError as exc:
            raise TTSModelError("Doubao TTS response is neither audio nor JSON.") from exc

        if result.get("code") not in (None, 0, 200, 3000):
            message = result.get("message") or result.get("error") or result
            raise TTSModelError(f"Doubao TTS returned an error: {message}")

        audio_data = result.get("data") or result.get("audio")
        if isinstance(audio_data, dict):
            audio_data = audio_data.get("audio") or audio_data.get("data")
        if not isinstance(audio_data, str) or not audio_data:
            raise TTSModelError("Doubao TTS response did not contain base64 audio data.")

        try:
            return base64.b64decode(audio_data)
        except ValueError as exc:
            raise TTSModelError("Doubao TTS response contains invalid base64 audio.") from exc
