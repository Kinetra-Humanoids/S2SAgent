# Copyright (C) 2026
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import base64
import io
import logging
import os
import uuid
from typing import Any

import numpy as np
import requests
from numpy.typing import NDArray
from scipy.io import wavfile

from rai_s2s.asr.models import BaseTranscriptionModel


class DoubaoASR(BaseTranscriptionModel):
    """Volcengine/Doubao BigASR flash transcription model."""

    def __init__(
        self,
        model_name: str,
        sample_rate: int,
        language: str = "zh",
        *,
        url: str | None = None,
        app_key: str | None = None,
        access_key: str | None = None,
        api_key: str | None = None,
        auth_mode: str = "auto",
        resource_id: str | None = None,
        timeout: float = 30.0,
    ):
        super().__init__(model_name, sample_rate, language)
        self.url = (
            url
            or os.getenv("DOUBAO_ASR_URL")
            or "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
        )
        self.app_key = app_key or os.getenv("DOUBAO_ASR_APP_KEY") or os.getenv("DOUBAO_APP_ID")
        self.access_key = access_key or os.getenv("DOUBAO_ASR_ACCESS_KEY") or os.getenv("DOUBAO_TOKEN")
        self.api_key = api_key or os.getenv("DOUBAO_ASR_API_KEY")
        self.auth_mode = auth_mode or os.getenv("DOUBAO_ASR_AUTH_MODE") or "auto"
        self.resource_id = (
            resource_id
            or os.getenv("DOUBAO_ASR_RESOURCE_ID")
            or "volc.bigasr.auc_turbo"
        )
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        if self.auth_mode not in {"auto", "new", "old"}:
            raise ValueError("DoubaoASR auth_mode must be one of: auto, new, old.")
        if self.auth_mode == "new" and not (self.api_key or self.app_key):
            raise ValueError("DoubaoASR new auth mode requires DOUBAO_ASR_API_KEY.")
        if self.auth_mode == "old" and not (self.app_key and self.access_key):
            raise ValueError(
                "DoubaoASR old auth mode requires both DOUBAO_ASR_APP_KEY "
                "and DOUBAO_ASR_ACCESS_KEY."
            )
        if self.auth_mode == "auto" and not self.api_key and not self.app_key:
            raise ValueError(
                "DoubaoASR requires DOUBAO_ASR_API_KEY or DOUBAO_ASR_APP_KEY. "
                "New console credentials only need an API key; old console "
                "credentials need app key plus access key."
            )

    def transcribe(self, data: NDArray[np.int16]) -> str:
        wav_bytes = self._to_wav_bytes(data)
        payload = {
            "user": {"uid": self.app_key or self.api_key or "rai_s2s"},
            "audio": {"data": base64.b64encode(wav_bytes).decode("utf-8")},
            "request": {
                "model_name": self.model_name,
                "enable_itn": True,
                "enable_punc": True,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
        }
        if self.auth_mode == "old":
            headers["X-Api-App-Key"] = self.app_key or ""
            headers["X-Api-Access-Key"] = self.access_key or ""
        elif self.auth_mode == "new":
            headers["X-Api-Key"] = self.api_key or self.app_key or ""
        elif self.api_key:
            headers["X-Api-Key"] = self.api_key
        elif self.access_key:
            headers["X-Api-App-Key"] = self.app_key or ""
            headers["X-Api-Access-Key"] = self.access_key or ""
        else:
            headers["X-Api-Key"] = self.app_key or ""

        response = requests.post(
            self.url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            message = response.headers.get("X-Api-Message", "")
            logid = response.headers.get("X-Tt-Logid", "")
            status_code = response.headers.get("X-Api-Status-Code", "")
            raise requests.HTTPError(
                "Doubao ASR request failed: "
                f"http_status={response.status_code}, "
                f"x_api_status_code={status_code}, "
                f"x_api_message={message}, "
                f"x_tt_logid={logid}, "
                f"body={response.text[:500]}"
            ) from exc
        result = response.json()
        transcription = self._extract_text(result)
        self.latest_transcription = transcription
        self.logger.info("transcription: %s", transcription)
        return transcription

    def _to_wav_bytes(self, data: NDArray[np.int16]) -> bytes:
        with io.BytesIO() as buffer:
            wavfile.write(buffer, self.sample_rate, data.astype(np.int16))
            return buffer.getvalue()

    def _extract_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [self._extract_text(item) for item in value]
            return "".join(part for part in parts if part)
        if not isinstance(value, dict):
            return ""

        for key in ("text", "utterance", "transcript", "result_text"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()

        result = value.get("result")
        if isinstance(result, dict):
            text = result.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
            utterances = result.get("utterances")
            if isinstance(utterances, list):
                parts = [self._extract_text(item) for item in utterances]
                text = "".join(part for part in parts if part).strip()
                if text:
                    return text

        for nested in value.values():
            text = self._extract_text(nested)
            if text:
                return text
        return ""
