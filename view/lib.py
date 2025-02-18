from pathlib import Path
import numpy as np
import soundfile as sf
import sounddevice as sd
from typing import Dict
from view.abstract import AbstractView


class NoView(AbstractView):
    def _validate_speed(self, speed_input: float) -> float:
        """Ensures speed is within the valid range."""
        return max(0.5, min(2.0, speed_input))

    def get_params(self, voice: str, speed: float, text: str):
        """Returns the provided parameters without prompting for user input."""
        return voice, self._validate_speed(speed), text

    def show_generated_segment(self, gs, ps) -> Dict[str, str]:
        """Logs the generated segment and phonemes (if needed)."""
        return {"gs": gs, "ps": ps}

    def save_audio(self, audio: np.ndarray, sample_rate: int, output_path: Path):
        """Saves audio to a file without retry logic."""
        try:
            sf.write(output_path, audio, sample_rate)
        except Exception as e:
            raise RuntimeError(f"Error saving audio: {e}")

    def prompt_play_audio(self) -> bool:
        """Always returns False since this is a library class."""
        return True

    def play_audio(self, audio: np.ndarray, sample_rate: int):
        """Plays audio without user confirmation."""
        try:
            sd.play(audio, sample_rate, blocksize=2048)
            sd.wait()
        except Exception as e:
            raise RuntimeError(f"Error playing audio: {e}")

    def get_audio(self, path: str):
        """Reads audio from a file and returns its data and sample rate."""
        data, samplerate = sf.read(path, dtype="float32")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.shape[0] < data.shape[1]:
            data = data.T
        return data, samplerate

    def save_audio_with_retry(
        self, audio: np.ndarray, sample_rate: int, output_path: Path
    ):
        """Calls save_audio directly without retrying."""
        self.save_audio(audio, sample_rate, output_path)

    def show_no_audio_generated(self):
        """Placeholder for handling cases where no audio is generated."""
        return -1

    def show_available_voices(self, voices: list):
        """Handles voice listing internally without printing."""
        return voices  # Can be used programmatically

    def show_exit(self):
        """Handles exit message without printing."""
        pass

    def get_menu_selection(self, choices) -> str:
        """Returns a default or first available choice instead of prompting."""
        return next(iter(choices.keys()), "")

    def show_invalid_choice(self) -> int:
        """Handles invalid choices internally."""
        return -1
