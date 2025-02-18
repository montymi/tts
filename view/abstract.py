from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class AbstractView(ABC):
    def set_voices(self, voices: list):
        self.voices = voices

    @abstractmethod
    def get_params(self, voice: str, speed: float, text: str, debug: bool):
        """
        Get user inputs for voice, speed, and text.
        """
        pass

    @abstractmethod
    def show_generated_segment(self, gs, ps):
        """
        Display information about a generated speech segment.
        """
        pass

    @abstractmethod
    def save_audio_with_retry(
        self, audio: np.ndarray, sample_rate: int, output_path: Path
    ):
        """
        Save the generated audio with retry logic in case of failures.
        """
        pass

    @abstractmethod
    def prompt_play_audio(self) -> bool:
        """
        Prompt the user to play the generated audio.
        """
        pass

    @abstractmethod
    def get_audio(self, path: str):
        """
        Get audio data from a file.
        """
        pass

    @abstractmethod
    def play_audio(self, audio: np.ndarray, sample_rate: int):
        """
        Play audio.
        """
        pass

    @abstractmethod
    def show_no_audio_generated(self):
        """
        Notify the user that no audio was generated.
        """
        pass

    @abstractmethod
    def show_available_voices(self, voices: list):
        """
        Display the available voices.
        """
        pass

    @abstractmethod
    def show_exit(self):
        """
        Display exit message.
        """
        pass

    @abstractmethod
    def get_menu_selection(self) -> str:
        """
        Get the user's menu selection.
        """
        pass

    @abstractmethod
    def show_invalid_choice(self):
        """
        Notify the user of an invalid menu choice.
        """
        pass
