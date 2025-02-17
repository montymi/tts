print("Starting TTS demo...")
try:
    import torch
    from typing import List
    from typing import cast
    from models import build_model, generate_speech, list_available_voices
    from tqdm.auto import tqdm
    from numbers import Number
    from pathlib import Path
    from prompt_toolkit.shortcuts import radiolist_dialog, message_dialog, input_dialog
    import soundfile as sf
    import numpy as np
    import time
    import sys
    import os
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(
        "Please ensure you have the required modules installed, or you have activated your virtual environment."
    )
    exit()
except KeyboardInterrupt:
    print("\nCancelling initialization...")
    exit()

# Constants
SAMPLE_RATE = 24000
DEFAULT_MODEL_PATH = "kokoro-v1_0.pth"
DEFAULT_OUTPUT_FILE = "output.wav"
DEFAULT_LANGUAGE = "a"  # 'a' for American English, 'b' for British English
DEFAULT_TEXT = "Hello, welcome to this text-to-speech test."

# Configure tqdm for better Windows console support
tqdm.monitor_interval = 0


def print_menu():
    """Display a fullscreen menu using prompt_toolkit."""
    options = [
        ("1", "List available voices"),
        ("2", "Generate speech"),
        ("3", "Exit"),
    ]

    result = radiolist_dialog(
        title="Kokoro TTS Menu",
        text="Use arrow keys to navigate and Enter to select an option:",
        values=options,
    ).run()

    return result  # Returns selected option as a string or None if canceled


def select_voice(voices: List[str]) -> str:
    """Interactive voice selection."""
    options = [(s.lower().replace(" ", "_"), s) for s in voices]

    while True:
        result = radiolist_dialog(
            title="Select an Option",
            text="Choose one of the available options:",
            values=options,
        ).run()

        if result:
            print(f"\n✅ You selected: {result}\n")
            return result  # Exit loop when selection is made
        else:
            print("\n❌ No option selected. Please try again.\n")


def get_text_input() -> str:
    """Get text input from user using a fullscreen input dialog."""
    text = input_dialog(
        title="Text Input",
        text="Enter the text you want to convert to speech (leave empty for default):",
    ).run()

    return text.strip() if text else DEFAULT_TEXT


def get_speed() -> float:
    """Get speech speed from user."""
    while True:
        try:
            speed = input_dialog(
                title="Speech Speed",
                text="Enter speech speed (0.5 - 2.0, default 1.0):",
            ).run()
            if not speed:
                speed = 1.0
            try:
                speed = float(speed)  # Convert to float (or int if needed)
            except ValueError:
                raise TypeError(f"Value '{speed}' cannot be converted to a Number")
            if 0.5 <= speed <= 2.0:
                return speed
            else:
                input_dialog(
                    title="Error",
                    text="⚠️ Speed must be between 0.5 and 2.0. Try again.",
                ).run()
        except ValueError:
            input_dialog(
                title="Error", text="⚠️ Please enter a valid number. Try again."
            ).run()


def save_audio_with_retry(
    audio_data: np.ndarray,
    sample_rate: int,
    output_path: Path,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> bool:
    """
    Attempt to save audio data to file with retry logic.
    Returns True if successful, False otherwise.
    """
    for attempt in range(max_retries):
        try:
            sf.write(output_path, audio_data, sample_rate)
            return True
        except Exception:
            if attempt < max_retries - 1:
                print(f"\nFailed to save audio (attempt {attempt + 1}/{max_retries})")
                print(
                    "The output file might be in use by another program (e.g., media player)."
                )
                print(f"Please close any programs that might be using '{output_path}'")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"\nError: Could not save audio after {max_retries} attempts.")
                print(
                    f"Please ensure '{output_path}' is not open in any other program and try again."
                )
                return False
    return False


def main(debug: bool = False) -> None:
    model = None
    try:
        # Set up device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Build model
        print("\nInitializing model...", end="")
        if not debug:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
        model = build_model(DEFAULT_MODEL_PATH, device)
        if not debug:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        if not model:
            print("ERROR: Failed to initialize model")
            raise (KeyboardInterrupt)
        else:
            print("SUCCESS: TTS Ready")

        while True:
            choice = print_menu()

            if choice == "1":
                # List voices
                voices = (
                    list_available_voices()
                )  # Assume this returns a list of voice names

                if not voices:
                    message_dialog(
                        title="Available Voices", text="❌ No voices available."
                    ).run()
                    return

                # Format the voice list for display
                voices_text = "\n".join(f"🎤 {voice}" for voice in voices)

                # Show the dialog
                message_dialog(title="Available Voices", text=voices_text).run()
            elif choice == "2":
                # Generate speech
                voices = list_available_voices()
                if not voices:
                    print("No voices found! Please check the voices directory.")
                    continue

                # Get user inputs
                voice = select_voice(voices)
                text = get_text_input()
                speed = get_speed()

                print(f"\nGenerating speech for: '{text}'")
                print(f"Using voice: {voice}")
                print(f"Speed: {speed}x")

                # Generate speech
                all_audio = []
                generator = model(
                    text, voice=f"voices/{voice}.pt", speed=speed, split_pattern=r"\n+"
                )

                with tqdm(desc="Generating speech") as pbar:
                    for gs, ps, audio in generator:
                        if audio is not None:
                            if isinstance(audio, np.ndarray):
                                audio = torch.from_numpy(audio).float()
                            all_audio.append(audio)
                            pbar.update(1)

                all_audio, ps, gs = generate_speech(model, text, voice, device, speed)

                print(f"\nGenerated segment: {gs}")
                print(f"Phonemes: {ps}")

                # Save audio
                if all_audio:
                    final_audio = torch.cat(all_audio, dim=0)
                    output_path = Path(DEFAULT_OUTPUT_FILE)
                    if save_audio_with_retry(
                        final_audio.numpy(), SAMPLE_RATE, output_path
                    ):
                        print(f"\nAudio saved to {output_path.absolute()}")
                else:
                    print("Error: Failed to generate audio")

            elif choice == "3":
                print("\nGoodbye!")
                break

            else:
                print("\nInvalid choice. Please try again.")

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback

        traceback.print_exc()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        # Cleanup
        try:
            if locals().get("model") is not None:
                del model
            torch.cuda.empty_cache()
        except NameError:
            pass  # Model was never defined, so nothing to delete

        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        debug = False
        d = input(
            "Press Enter to start the TTS demo...type d, D, debug, or DEBUG to enable debug mode: "
        )
        if d.lower() in ["d", "debug"]:
            debug = True
        main(debug)
    except KeyboardInterrupt:
        print("Cancelling initialization...")
