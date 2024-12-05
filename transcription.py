import whisper
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Transcribe an audio file using OpenAI's Whisper model.")
parser.add_argument("audio_file", type=str, help="Path to the audio file to transcribe.")

# Parse the arguments
args = parser.parse_args()

# Load the Whisper model
model = whisper.load_model("base")  # You can change "base" to other sizes like "tiny", "small", etc.

# Transcribe the audio file
result = model.transcribe(args.audio_file, language="en")

# Print the transcription
print("Transcription:")
print(result["text"])

# Optionally, save the transcription to a file
output_file = args.audio_file + "_transcription.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(result["text"])

print(f"\nTranscription saved to {output_file}")


# pip install openai-whisper
# pip uninstall torch
# pip install torch torchvision torchaudio
# Whisper requires ffmpeg to handle audio processing. 

# tiny: Fastest, but less accurate.
# base: Slightly more accurate, still fast.
# small: Good balance between speed and accuracy.
# medium: More accurate, slower.
# large: Most accurate, but slowest.

# python t.py 1.mp3
