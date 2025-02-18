from models import build_model, generate_speech, list_available_voices
from pathlib import Path
import torch
import logging
import sys
import os

SAMPLE_RATE = 24000
DEFAULT_MODEL_PATH = "kokoro-v1_0.pth"
DEFAULT_OUTPUT_FILE = "output.wav"
DEFAULT_LANGUAGE = "a"  # 'a' for American English, 'b' for British English
DEFAULT_TEXT = "Hello, welcome to this text-to-speech test."


class Controller:
    def __init__(
        self,
        view,
        debug: bool = False,
        speed: float = 1.0,
        output_file: str = DEFAULT_OUTPUT_FILE,
        text: str = DEFAULT_TEXT,
    ):
        self.OUTPUT = output_file
        self.view = view
        self.text = text
        self.speed = speed
        self.debug = debug
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.voices = []
        self.choices = {
            "list": self.handle_list_voices,
            "generate": self.handle_generate_speech,
            "play": self.handle_play_audio,
            "exit": self.handle_exit,
        }

    def __init_model__(self):
        logging.debug(f"Building model to {self.OUTPUT}")
        if not self.debug:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
        model = build_model(self.OUTPUT, self.device)
        if not self.debug:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        if not model:
            logging.error("Failed to initialize model")
            raise (KeyboardInterrupt)
        return model

    def handle_play_audio(self):
        data, samplerate = self.view.get_audio(self.OUTPUT)
        self.view.play_audio(data, samplerate)

    def handle_generate_speech(self, text: str = "", quiet: bool = False):
        if self.voices == [] or self.model is None:
            print(
                "App not properly initialized. Voices and model found to be:",
                self.voices,
                self.model,
            )
            raise (KeyboardInterrupt)

        # Get user inputs
        self.voice, self.speed, self.text = self.view.get_params(
            self.voice, self.speed, self.text
        )
        if text != "":
            self.text = text
        # Generate speech
        all_audio, ps, gs = generate_speech(
            self.model, self.text, self.voice, self.speed
        )

        # Save audio
        if all_audio:
            final_audio = torch.cat(all_audio, dim=0)
            self.view.show_generated_segment(gs, ps)
            if self.view.prompt_play_audio() and not quiet:
                self.view.play_audio(final_audio, SAMPLE_RATE)
            output_path = Path(self.OUTPUT)
            self.view.save_audio_with_retry(
                final_audio.numpy(), SAMPLE_RATE, output_path
            )
        else:
            self.view.show_no_audio_generated()

    def handle_list_voices(self):
        return self.view.show_available_voices(self.voices)

    def handle_exit(self):
        self.view.show_exit()
        raise (KeyboardInterrupt)

    def handle_menu(self):
        choice = self.view.get_menu_selection(self.choices)
        if choice in self.choices.keys():
            return choice
        else:
            self.view.show_invalid_choice()
            return self.handle_menu()

    def handle_set_voice(self, voice: str):
        self.voice = voice if voice in self.voices else self.voices[0]

    def load(self):
        self.model = self.__init_model__()
        self.voices = list_available_voices()
        self.view.set_voices(self.voices)
        self.voice = self.voices[0]

    def start(self):
        try:
            while True:
                choice = self.handle_menu()
                self.choices[choice]()
        except KeyboardInterrupt:
            logging.info("Exiting...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Welcome to TTS")
    from view.cli import CLIView

    controller = Controller(CLIView())

    logging.debug("Controller loading...")
    controller.load()

    logging.debug("Controller starting...")
    controller.start()
