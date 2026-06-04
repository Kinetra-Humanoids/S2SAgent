# Copyright (C) 2026 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread
from typing import Optional, TypedDict

from numpy._typing import NDArray
from pydub import AudioSegment


class ThreadData(TypedDict):
    thread: Thread
    event: Event
    transcription: str
    joined: bool


@dataclass
class PlayData:
    playing: bool = False
    current_segment: Optional[AudioSegment] = None
    data: Optional[NDArray] = None
    channels: int = 1
    current_frame: int = 0
