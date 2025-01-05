import yt_dlp
import whisper
import sys
import os
from whisper.utils import get_writer
import logging

def transcribe(link):
    downloaded_filename = downloader.prepare_filename(downloader.extract_info(link, download=False))
    filename = downloaded_filename.split('.')[0]
    if os.path.exists(f'output/{filename}'):
        logging.warning(f'Output file already exists, skipping {link}')
        return
    downloader.download(link)
    result = model.transcribe(downloaded_filename)
    # Write to output directory
    get_writer('txt', 'output')(result, f'{filename}.txt')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Correct usage: python3 transcribe.py [video_link1], [video_link2], ...')
    else:
        model = whisper.load_model("tiny.en")
        downloader = yt_dlp.YoutubeDL(params={'format':'ba', 'outtmpl' : {'default': f"output/%(upload_date)s_%(title)s.mp3"}})
        for x in sys.argv[1::]:
           transcribe(x)

