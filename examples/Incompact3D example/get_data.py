import gdown
import hashlib
import os

from zipfile import ZipFile


def main():

    # Get file from my personal google drive which contains dataset for use in this tutorial
    # Link can be found here: https://drive.google.com/file/d/1fyefz3EyNooZjE3ugD2J-vOGkfXzlCQb/view?usp=drive_link

    # download zip file
    out = 'incompact_channel.zip'
    url = 'https://drive.google.com/file/d/1fyefz3EyNooZjE3ugD2J-vOGkfXzlCQb/view?usp=drive_link'
    gdown.download(output=out,
                   url=url,
                   fuzzy=True)

    # Check sha256sum
    correct_hash = 'e841b7c322233b85eb19e5f8a9ea80a49acf18edf7475b4e387fd6d7edb9e85d'
    with open(out, 'rb', buffering=0) as f:
        file_hash = hashlib.file_digest(f, 'sha256').hexdigest()

    if file_hash != correct_hash:
        raise Exception("Has of downloaded file does not match")

    # extract zip file
    with ZipFile(out, 'r') as zip:
        zip.extractall()

    os.remove(out)


if __name__ == '__main__':
    main()