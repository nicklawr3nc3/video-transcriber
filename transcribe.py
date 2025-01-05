import yt_dlp
import whisper
import sys
import os
from whisper.utils import get_writer
import logging
import argparse

def download(link):
    if os.path.exists(filepath):
        logging.warning(f'Output file already exists, skipping {link}')
        return False
    downloader.download(link)
    return True

def transcribe(link):
    filename = filepath.split('.')[0]
    result = model.transcribe(filepath)
    # Write to output directory. vtt includes timestamps of text
    get_writer('vtt', 'output')(result, filepath)
    # Include link to video in results
    with open(f'{filename}.vtt', 'a') as f:
        f.write(link)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple argument parser")
    # Add arguments
    parser.add_argument("-f", '--format', help="Output file format")
    parser.add_argument("-m", '--model', help="OpenAI Whisper model to use")
    parser.add_argument("additional_args", nargs='*', help="Additional arguments")
    # Parse the arguments
    args = parser.parse_args()

    if len(sys.argv) < 2:
        print('Correct usage: python3 transcribe.py [ARGS] [video_link1], [video_link2], ...')
    else:
        model = whisper.load_model(args.model or "tiny.en")
        downloader = yt_dlp.YoutubeDL(params={'format': args.format or 'ba', 'outtmpl' : {'default': f"output/%(upload_date)s_%(title)s.mp3"}})
        print(args)
        for x in args.additional_args:
            filepath = downloader.prepare_filename(downloader.extract_info(x, download=False))
            # Only transcribe for default format
            if download(x) and not args.format:
                transcribe(x)

