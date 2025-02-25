import requests
from pathlib import Path
from tqdm import tqdm

def downloader(fname, url, message, force_download=False):
    if not Path(fname).is_file() or force_download:
        print(message)
        # print('Downloading {0:s} tables from PAdova TRieste Stellar Evolutionary Code.'.\
        #       format(kind))
        # print('https://stev.oapd.inaf.it/PARSEC/tracks_v2_VMS.html')
        response = requests.get(url, stream=True)
        if not response.ok:
            raise Exception('Failed to download file. Check URL.')

        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        tqdm_args=dict(desc='Downloading', total=total_size, unit='B',
                       unit_scale=True, unit_divisor=1024)
        with open(fname, 'wb') as f, tqdm(**tqdm_args) as bar:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    bar.update(len(chunk))
