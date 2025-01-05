import yt_dlp
import whisper
import sys
import os
from whisper.utils import get_writer
import logging

def transcribe(link):
    try:
        downloaded_filename = downloader.prepare_filename(downloader.extract_info(link, download=False))
        dirname = downloaded_filename.split('.')[0]
        os.makedirs(dirname)
    except:
        logging.warning(f'Output directory already exists, skipping {link}')
        return
    downloader.download(link)
    result = model.transcribe(downloaded_filename)
    # Write to output directory
    get_writer('txt', dirname)(result, 'transcript.txt')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Correct usage: python3 transcribe.py [video_link1], [video_link2], ...')
    else:
        model = whisper.load_model("tiny.en")
        downloader = yt_dlp.YoutubeDL(params={'format':'ba', 'outtmpl' : {'default': f"%(title)s_%(upload_date)s.mp3"}})
        for x in sys.argv[1::]:
           transcribe(x)

