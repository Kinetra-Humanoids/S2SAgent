# Copyright (C) 2024 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .tts.models import KokoroTTS, OpenTTS

try:
	from .tts.agents import TextToSpeechAgent

	__all__ = ["ElevenLabsTTS", "KokoroTTS", "OpenTTS", "TextToSpeechAgent"]
except Exception:
	# ROS2 is an optional dependency for rai_s2s; TextToSpeechAgent requires ROS2
	# connectors at import time.
	__all__ = ["ElevenLabsTTS", "KokoroTTS", "OpenTTS"]


def __getattr__(name: str):
	if name == "ElevenLabsTTS":
		from .tts.models import ElevenLabsTTS

		return ElevenLabsTTS
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
