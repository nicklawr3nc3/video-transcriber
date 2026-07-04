import yt_dlp
from faster_whisper import WhisperModel
import ctranslate2
import sys
import os
import logging
import argparse
import subprocess
from datetime import datetime, timezone
from dotenv import load_dotenv
import numpy as np
import requests

LIRIK_USER_ID = '23161357'

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def resolve_device(requested):
    cuda_available = ctranslate2.get_cuda_device_count() > 0
    if requested == 'auto':
        return 'cuda' if cuda_available else 'cpu'
    if requested == 'cuda' and not cuda_available:
        logging.warning('CUDA requested but unavailable; falling back to CPU')
        return 'cpu'
    return requested

# def resolve_compute_type(device):
#     return 'float16' if device == 'cuda' else 'int8'

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f'{hours:02d}:{minutes:02d}:{secs:06.3f}'

def get_audio_duration(filepath):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', filepath],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())

def load_audio_chunk(filepath, start, duration, sampling_rate=16000):
    result = subprocess.run(
        ['ffmpeg', '-hide_banner', '-loglevel', 'error',
         '-ss', str(start), '-t', str(duration),
         '-i', filepath,
         '-f', 's16le', '-acodec', 'pcm_s16le',
         '-ac', '1', '-ar', str(sampling_rate), '-'],
        capture_output=True, check=True,
    )
    if not result.stdout:
        return np.array([], dtype=np.float32)
    return np.frombuffer(result.stdout, np.int16).astype(np.float32) / 32768.0

def write_vtt_segment(f, segment, time_offset=0):
    start = format_timestamp(segment.start + time_offset)
    end = format_timestamp(segment.end + time_offset)
    f.write(f'{start} --> {end}\n')
    f.write(segment.text.strip() + '\n\n')

def get_twitch_oauth():
    load_dotenv()
    client_id = os.getenv('client_id')
    client_secret = os.getenv('client_secret')
    response = requests.post(
        'https://id.twitch.tv/oauth2/token',
        f'client_id={client_id}&client_secret={client_secret}&grant_type=client_credentials',
        headers={'Content-Type': 'application/x-www-form-urlencoded'},
    ).json()
    return response['access_token'], client_id

def is_streamer_live(oauth, client_id, user_id=LIRIK_USER_ID):
    response = requests.get(
        'https://api.twitch.tv/helix/streams',
        params={'user_id': user_id},
        headers={'Authorization': f'Bearer {oauth}', 'Client-Id': client_id},
    ).json()
    return bool(response.get('data'))

def is_today_utc(date_str):
    if not date_str:
        return False
    today = datetime.now(timezone.utc).date()
    if len(date_str) >= 10 and date_str[4] == '-':
        created = datetime.fromisoformat(date_str[:10]).date()
    else:
        created = datetime.strptime(date_str[:8], '%Y%m%d').date()
    return created == today

def cleanup_partial_download(path):
    for suffix in ('', '.part', '.ytdl'):
        partial = f'{path}{suffix}' if suffix else path
        if os.path.exists(partial):
            os.remove(partial)

def download(link):
    if os.path.exists(filepath):
        logging.info(f'Audio file already exists, skipping download: {filepath}')
        return
    try:
        downloader.download(link)
    except yt_dlp.utils.DownloadError as e:
        if 'Initialization fragment found after media fragments' not in str(e):
            raise
        logging.warning('HLS init fragment error; retrying with ffmpeg downloader')
        cleanup_partial_download(filepath)
        ffmpeg_params = {**yt_dlp_params, 'downloader': 'ffmpeg'}
        yt_dlp.YoutubeDL(ffmpeg_params).download([link])

def transcribe(link, chunk_length=1800):
    filename = os.path.splitext(filepath)[0]
    vtt_path = f'{filename}.vtt'
    txt_path = f'{filename}.txt'
    if os.path.exists(vtt_path) and os.path.exists(txt_path):
        logging.info(f'Transcription already exists, skipping: {vtt_path}')
        return

    duration = get_audio_duration(filepath)
    transcribe_opts = {'beam_size': 1, 'condition_on_previous_text': False}
    txt_parts = []
    chunk_start = 0.0

    with open(vtt_path, 'w') as vtt_f, open(txt_path, 'w') as txt_f:
        vtt_f.write('WEBVTT\n\n')
        while chunk_start < duration:
            chunk_dur = min(chunk_length, duration - chunk_start)
            logging.info(
                'Transcribing chunk %s -> %s',
                format_timestamp(chunk_start),
                format_timestamp(chunk_start + chunk_dur),
            )
            audio = load_audio_chunk(filepath, chunk_start, chunk_dur)
            if len(audio) == 0:
                break
            segments, _info = model.transcribe(audio, **transcribe_opts)
            for segment in segments:
                write_vtt_segment(vtt_f, segment, chunk_start)
                txt_parts.append(segment.text.strip())
            chunk_start += chunk_dur
        txt_f.write(' '.join(txt_parts) + '\n')

    with open(vtt_path, 'a') as f:
        f.write(link)
    with open(txt_path, 'a') as f:
        f.write(f'\n\nVideo link: {link}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple argument parser")
    # Add arguments
    parser.add_argument("-f", '--format', help="Output file format")
    parser.add_argument("-F", '--file', help="File to read from")
    parser.add_argument("-l", '--latest', type=int, help="Pull lirik's latest N VODs (default: 10)", nargs='?', const=10, default=None)
    parser.add_argument("-m", '--model', help="Whisper model to use. Options are: tiny.en, base.en, small.en, medium.en, large-v3, turbo")
    parser.add_argument("-D", '--device', default='auto', choices=['auto', 'cuda', 'cpu'], help="Device for Whisper inference (default: auto)")
    parser.add_argument("-d", '--duration', type=int, help="Duration in seconds to download (default: download entire video)")
    parser.add_argument("-c", '--chunk-length', type=int, default=1800, help="Transcribe audio in chunks of N seconds to limit memory use (default: 1800)")
    parser.add_argument("additional_args", nargs='*', help="Additional arguments")
    # Parse the arguments
    args = parser.parse_args()

    if len(sys.argv) < 2:
        print('Correct usage: python3 transcribe.py [ARGS] [video_link1], [video_link2], ...')
    else:
        device = resolve_device(args.device)
        compute_type = 'int8'
        if device == 'cuda':
            logging.info(f'Using GPU ({ctranslate2.get_cuda_device_count()} CUDA device(s), compute_type={compute_type})')
        else:
            logging.info(f'Using CPU for Whisper inference (compute_type={compute_type})')
        model = WhisperModel(args.model or 'small.en', device=device, compute_type=compute_type)
        yt_dlp_params = {
            'format': args.format or 'ba',
            'outtmpl': {'default': f"output/%(upload_date)s_%(title)s.mp3"},
        }
        # Set download ranges
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
        stream_is_live = False
        if args.latest is not None:
            oauth, client_id = get_twitch_oauth()
            stream_is_live = is_streamer_live(oauth, client_id)
            count = args.latest if args.latest > 0 else 10
            response = requests.get(
                'https://api.twitch.tv/helix/videos',
                params={'user_id': LIRIK_USER_ID, 'sort_by': 'time', 'type': 'archive', 'first': str(count)},
                headers={'Authorization': f'Bearer {oauth}', 'Client-Id': client_id},
            ).json()
            videos = []
            for video in response['data']:
                if stream_is_live and is_today_utc(video['created_at']):
                    logging.info("Skipping today's VOD (stream still live): %s", video['title'])
                    continue
                videos.append(video['url'])
        for x in videos:
            info = downloader.extract_info(x, download=False)
            filepath = downloader.prepare_filename(info)
            if stream_is_live and is_today_utc(info.get('upload_date', '')):
                logging.info("Skipping today's VOD (stream still live): %s", info.get('title', x))
                continue
            # Only transcribe for default format
            download(x)
            if not args.format:
                transcribe(x, chunk_length=args.chunk_length)
