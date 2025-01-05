import yt_dlp
import whisper
import sys
import os

def transcribe(link):
    downloaded_filename = downloader.prepare_filename(downloader.extract_info(link, download=False))
    downloader.download(link)
    result = model.transcribe(downloaded_filename)
    # Save the transcription to a file in your desired directory
    dirname = downloaded_filename.split('.')[0]
    os.makedirs(dirname)
    with open(os.path.join(dirname, "transcription.txt"), "w") as f:
        f.write(result['text'])

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Correct usage: python3 transcribe.py [video_link1], [video_link2], ...')
    else:
        processes = []
        model = whisper.load_model("tiny.en")
        downloader = yt_dlp.YoutubeDL(params={'format':'ba', 'outtmpl' : {'default': f"%(upload_date)s.mp3"}})
        for x in sys.argv[1::]:
           transcribe(x)

