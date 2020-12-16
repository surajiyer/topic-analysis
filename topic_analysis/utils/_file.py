import os
import progressbar as pb
from typing import List
import urllib.request


def download(url: str, save_path: str):
    """
    Download file from URL.

    Parameters
    ----------
    url : str
        URL of the file to download.

    save_path : str
        file path (incl. file name) to save to.
    """
    # Create progressbar
    widgets = ['Downloaded: ', pb.Percentage(),
               ' ', pb.Bar(marker=pb.RotatingMarker()),
               ' ', pb.ETA(),
               ' ', pb.FileTransferSpeed()]
    pbar = pb.ProgressBar(widgets=widgets)

    def dl_progress(count, blockSize, totalSize):
        if pbar.max_value is None:
            pbar.max_value = totalSize
            pbar.start()
        pbar.update(min(count * blockSize, totalSize))

    # Create the save path if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Download the file
    print(f'Downloading file from {url}.')
    try:
        urllib.request.urlretrieve(url, save_path, reporthook=dl_progress)
    except Exception as e:
        print(e)
        print(
            'Unable to download file. Please download '
            'manually from the URL above and place it in '
            f'the following path {save_path}.')

    # Close progressbar
    pbar.finish()
