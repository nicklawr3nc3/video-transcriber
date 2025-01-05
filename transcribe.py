import yt_dlp
import whisper
import sys
import os
from whisper.utils import get_writer
import logging

def transcribe(link):
    filepath = downloader.prepare_filename(downloader.extract_info(link, download=False))
    filename = filepath.split('.')[0]
    if os.path.exists(filepath):
        logging.warning(f'Output file already exists, skipping {link}')
        return
    downloader.download(link)
    result = model.transcribe(filepath)
    # Write to output directory. vtt includes timestamps of text
    get_writer('vtt', 'output')(result, filepath)
    # Include link to video in results
    with open(f'{filename}.vtt', 'a') as f:
        f.write(link)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Correct usage: python3 transcribe.py [video_link1], [video_link2], ...')
    else:
        model = whisper.load_model("tiny.en")
        downloader = yt_dlp.YoutubeDL(params={'format':'ba', 'outtmpl' : {'default': f"output/%(upload_date)s_%(title)s.mp3"}})
        for x in sys.argv[1::]:
           transcribe(x)

