from prompt_toolkit.shortcuts import prompt, CompleteStyle
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.shortcuts import confirm
from pathlib import Path
import numpy as np
import soundfile as sf
import sounddevice as sd
from .abstract import AbstractView


class CLIView(AbstractView):
    def _get_speed_input(self, speed):
        while True:
            try:
                speed_input = float(prompt(f"Enter speed ({speed}): ") or speed)
                if 0.5 <= speed_input <= 2.0:
                    return speed_input
                else:
                    print("Please enter a value between 0.5 and 2.0.")
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    def get_params(self, voice: str, speed: float, text: str):
        voice_completer = WordCompleter(self.voices)
        voice = prompt(f"Enter voice ({voice}): ", completer=voice_completer) or voice
        speed = self._get_speed_input(speed)
        text = prompt(f"Enter text ({text}): ") or text
        return voice, speed, text

    def show_generated_segment(self, gs, ps):
        print(f"\nGenerated segment: {gs}\nPhonemes: {ps}")

    def save_audio_with_retry(
        self, audio: np.ndarray, sample_rate: int, output_path: Path
    ):
        try:
            sf.write(output_path, audio, sample_rate)
            print(f"Audio path: {output_path}\n")
        except Exception as e:
            print(f"Error saving audio: {e}\n")

    def prompt_play_audio(self) -> bool:
        return confirm("Play audio?")

    def get_audio(self, path: str):
        data, samplerate = sf.read(path, dtype="float32")
        if data.ndim == 1:  # mono
            data = data.reshape(-1, 1)
        elif data.shape[0] < data.shape[1]:
            data = data.T
        return data, samplerate

    def play_audio(self, audio: np.ndarray, sample_rate: int):
        try:
            sd.play(audio, sample_rate, blocksize=2048)
            sd.wait()
            print("")
        except Exception as e:
            print(f"Error playing audio: {e}")

    def show_no_audio_generated(self):
        print("No audio was generated.")

    def show_available_voices(self, voices: list):
        print("Available voices:")
        for voice in voices:
            print(f"- {voice}")

    def show_exit(self):
        print("Goodbye!")

    def get_menu_selection(self, choices) -> str:
        completer = WordCompleter(list(choices.keys()))
        choice = prompt(
            "Select an option: ",
            completer=completer,
            complete_style=CompleteStyle.MULTI_COLUMN,
        )
        return choice.lower()

    def show_invalid_choice(self):
        print("Invalid choice. Please select a valid option.")
