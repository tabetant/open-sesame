"""
capture_wav.py — Convert a JTAG UART audio dump to a .wav file.

Usage
-----
1. In PowerShell, redirect nios2-terminal output to a file:

       nios2-terminal.exe --instance 0 | Tee-Object -FilePath capture.txt

2. Run the board normally. When you want to capture what the mic just heard,
   press KEY[0] on the DE1-SoC. The board dumps the last 2 seconds of audio.

3. Once you see DUMP_END in the terminal, Ctrl+C the terminal, then run:

       python training/capture_wav.py capture.txt output.wav

4. Move output.wav into training/dataset/positive/ or training/dataset/negative/
   and retrain.

The script handles multiple dumps in one file — use --index to pick which one
(default: last dump in the file).
"""

import sys
import struct
import wave
import argparse

SAMPLE_RATE = 8000  # must match the board


def parse_dumps(path):
    """Return a list of sample lists, one per DUMP_START/DUMP_END block."""
    dumps = []
    current = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line == "DUMP_START":
                current = []
            elif line == "DUMP_END":
                if current is not None:
                    dumps.append(current)
                current = None
            elif current is not None:
                try:
                    current.append(int(line))
                except ValueError:
                    pass  # skip non-integer lines (e.g. status messages)
    return dumps


def save_wav(samples, out_path):
    with wave.open(out_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit
        wf.setframerate(SAMPLE_RATE)
        for s in samples:
            s = max(-32768, min(32767, s))
            wf.writeframes(struct.pack("<h", s))
    duration = len(samples) / SAMPLE_RATE
    print(f"Saved {len(samples)} samples ({duration:.2f} s) → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert JTAG audio dump to WAV")
    parser.add_argument("input",  help="Text file captured from nios2-terminal")
    parser.add_argument("output", help="Output .wav file path")
    parser.add_argument(
        "--index", type=int, default=-1,
        help="Which dump to extract if the file has multiple (default: last)"
    )
    args = parser.parse_args()

    dumps = parse_dumps(args.input)
    if not dumps:
        print("ERROR: No DUMP_START/DUMP_END block found in the file.")
        print("Make sure you pressed KEY[0] on the board while nios2-terminal was running.")
        sys.exit(1)

    print(f"Found {len(dumps)} dump(s) in {args.input}")
    samples = dumps[args.index]
    save_wav(samples, args.output)


if __name__ == "__main__":
    main()
