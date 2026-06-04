# Copyright (C) 2026
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import os
import time
from functools import lru_cache
from uuid import uuid4

from pydub import AudioSegment


class UnitreeG1AudioError(Exception):
    pass


@lru_cache(maxsize=4)
def _get_audio_client(network_interface: str):
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
    except ImportError as exc:
        raise UnitreeG1AudioError(
            "Unitree SDK is not installed. Install unitree_sdk2_python before "
            "enabling Unitree G1 audio."
        ) from exc

    if not network_interface:
        raise UnitreeG1AudioError(
            "Missing Unitree network interface. Set UNITREE_G1_NETWORK_INTERFACE "
            "or configure it in the Streamlit configurator."
        )

    ChannelFactoryInitialize(0, network_interface)
    client = AudioClient()
    client.SetTimeout(10.0)
    client.Init()
    return client


class UnitreeG1AudioPlayer:
    """Play TTS audio through the Unitree G1 AudioClient."""

    sample_rate = 16000
    channels = 1
    sample_width = 2

    def __init__(
        self,
        *,
        network_interface: str | None = None,
        app_name: str = "rai_s2s",
        chunk_size: int = 96000,
        stop_after_play: bool = False,
    ):
        self.network_interface = (
            network_interface or os.getenv("UNITREE_G1_NETWORK_INTERFACE", "")
        )
        self.app_name = app_name
        self.chunk_size = chunk_size
        self.stop_after_play = stop_after_play

    @property
    def client(self):
        return _get_audio_client(self.network_interface)

    def play(self, audio: AudioSegment) -> None:
        pcm_audio = (
            audio.set_frame_rate(self.sample_rate)
            .set_channels(self.channels)
            .set_sample_width(self.sample_width)
        )
        pcm_data = pcm_audio.raw_data
        stream_id = str(uuid4())

        for offset in range(0, len(pcm_data), self.chunk_size):
            chunk = pcm_data[offset : offset + self.chunk_size]
            code = self.client.PlayStream(self.app_name, stream_id, chunk)
            if code != 0:
                raise UnitreeG1AudioError(
                    f"Unitree G1 PlayStream failed with SDK return code: {code}"
                )
            time.sleep(min(1.0, len(chunk) / (self.sample_rate * self.sample_width)))

        if self.stop_after_play:
            self.stop()

    def stop(self) -> None:
        code = self.client.PlayStop(self.app_name)
        if code != 0:
            raise UnitreeG1AudioError(
                f"Unitree G1 PlayStop failed with SDK return code: {code}"
            )
