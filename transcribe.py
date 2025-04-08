import yt_dlp
import whisper
import sys
import os
from whisper.utils import get_writer
import logging
import argparse
from dotenv import load_dotenv
import requests

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
    parser.add_argument("-F", '--file', help="File to read from")
    parser.add_argument("-l", '--latest', help="Pull lirik's latest VOD", action='store_true')
    parser.add_argument("-m", '--model', help="OpenAI Whisper model to use. Options are: tiny.en, base.en, small.en, medium.en, large, turbo")
    parser.add_argument("-d", '--duration', type=int, help="Duration in seconds to download (default: download entire video)")
    parser.add_argument("additional_args", nargs='*', help="Additional arguments")
    # Parse the arguments
    args = parser.parse_args()

    if len(sys.argv) < 2:
        print('Correct usage: python3 transcribe.py [ARGS] [video_link1], [video_link2], ...')
    else:
        model = whisper.load_model(args.model or "base.en")
        yt_dlp_params = {
            'format': args.format or 'ba', 
            'outtmpl': {'default': f"output/%(upload_date)s_%(title)s.mp3"}
        }
        if args.duration:
            yt_dlp_params.update({
                'download_ranges': lambda info, ydl: [{'start_time': 0, 'end_time': args.duration}],
                'force_keyframes_at_cuts': True
            })
        downloader = yt_dlp.YoutubeDL(params=yt_dlp_params)
        videos = args.additional_args
        if args.file:
           with open(args.file, 'r') as f:
                videos = f.readlines()
        if args.latest:
            load_dotenv()
            oauth = requests.post('https://id.twitch.tv/oauth2/token', f"client_id={os.getenv('client_id')}&client_secret={os.getenv('client_secret')}&grant_type=client_credentials", headers={'Content-Type': 'application/x-www-form-urlencoded'}).json()['access_token']
            videos = [requests.get('https://api.twitch.tv/helix/videos', params={'user_id': '23161357', 'sort_by': 'time', 'type': 'archive', 'first': '1'}, headers={'Authorization' : f'Bearer {oauth}', 'Client-Id': os.getenv('client_id')}).json()['data'][0]['url']]
        for x in videos:
            filepath = downloader.prepare_filename(downloader.extract_info(x, download=False))
            # Only transcribe for default format
            if download(x) and not args.format:
                transcribe(x)

