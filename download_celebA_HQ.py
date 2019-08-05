import zipfile
import os
import requests
import argparse
import hashlib

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Download celebA-HQ helper')
parser.add_argument('path', type=str)
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination, chunk_size=32 * 1024):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        for chunk in tqdm(
                response.iter_content(chunk_size),
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=destination):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == '__main__':
    args = parser.parse_args()
    dirpath = args.path
    data_dir = 'celebA-HQ'

    global_data_dir = os.path.join(dirpath, data_dir)
    if not os.path.exists(global_data_dir):
        os.makedirs(global_data_dir)

    filenames = [
        'deltas00000.zip', 'deltas01000.zip', 'deltas02000.zip',
        'deltas03000.zip', 'deltas04000.zip', 'deltas05000.zip',
        'deltas06000.zip', 'deltas07000.zip', 'deltas08000.zip',
        'deltas09000.zip', 'deltas10000.zip', 'deltas11000.zip',
        'deltas12000.zip', 'deltas13000.zip', 'deltas14000.zip',
        'deltas15000.zip', 'deltas16000.zip', 'deltas17000.zip',
        'deltas18000.zip', 'deltas19000.zip', 'deltas20000.zip',
        'deltas21000.zip', 'deltas22000.zip', 'deltas23000.zip',
        'deltas24000.zip', 'deltas25000.zip', 'deltas26000.zip',
        'deltas27000.zip', 'deltas28000.zip', 'deltas29000.zip',
        'image_list.txt'
    ]

    drive_ids = [
        '0B4qLcYyJmiz0TXdaTExNcW03ejA', '0B4qLcYyJmiz0TjAwOTRBVmRKRzQ',
        '0B4qLcYyJmiz0TjNRV2dUamd0bEU', '0B4qLcYyJmiz0TjRWUXVvM3hZZE0',
        '0B4qLcYyJmiz0TjRxVkZ1NGxHTXc', '0B4qLcYyJmiz0TjRzeWlhLVJIYk0',
        '0B4qLcYyJmiz0TjVkYkF4dTJRNUk', '0B4qLcYyJmiz0TjdaV2ZsQU94MnM',
        '0B4qLcYyJmiz0Tksyd21vRmVqamc', '0B4qLcYyJmiz0Tl9wNEU2WWRqcE0',
        '0B4qLcYyJmiz0TlBCNFU3QkctNkk', '0B4qLcYyJmiz0TlNyLUtOTzk3QjQ',
        '0B4qLcYyJmiz0Tlhvdl9zYlV4UUE', '0B4qLcYyJmiz0TlpJU1pleF9zbnM',
        '0B4qLcYyJmiz0Tm5MSUp3ZTZ0aTg', '0B4qLcYyJmiz0TmRZTmZyenViSjg',
        '0B4qLcYyJmiz0TmVkVGJmWEtVbFk', '0B4qLcYyJmiz0TmZqZXN3UWFkUm8',
        '0B4qLcYyJmiz0TmhIUGlVeE5pWjg', '0B4qLcYyJmiz0TnBtdW83OXRfdG8',
        '0B4qLcYyJmiz0TnJQSS1vZS1JYUE', '0B4qLcYyJmiz0TzBBNE8xbFhaSlU',
        '0B4qLcYyJmiz0TzZySG9IWlZaeGc', '0B4qLcYyJmiz0U05ZNG14X3ZjYW8',
        '0B4qLcYyJmiz0U0YwQmluMmJuX2M', '0B4qLcYyJmiz0U0lYX1J1Tk5vMjQ',
        '0B4qLcYyJmiz0U0tBanQ4cHNBUWc', '0B4qLcYyJmiz0U1BRYl9tSWFWVGM',
        '0B4qLcYyJmiz0U1BhWlFGRXc1aHc', '0B4qLcYyJmiz0U1pnMEI4WXN1S3M',
        '0B4qLcYyJmiz0U25vdEVIU3NvNFk'
    ]

    sha = [
        '9d8da3b6e6d8088524acacb4839c4bcce0fb95ed',
        'b54650a4954185198cfa53a0f1c7d3b5e247d353',
        '4252adb3ccd9d761b5ccf6a6bd5a67a88e5406a1',
        'bf7cb67e81a4bf6d6eea42ac2a4fdf67b3a7e0b9',
        '6afe57f3deb2bd759bbc41618f650cd5459b9e23',
        '64978b1a7f06ea83dd886b418af7575a02acded5',
        'f34caf938a06be84a570f579c2556d24c2458347',
        '79ef1c3db2ff4c1d31c7c5bf66493a14c7a1b5cb',
        '0c062a7809f7092101c9fe673e72d8bfd1e098b5',
        'd52635cf9c90a68da9f6337d247e1719959b2515',
        '1485ec0b67d1f30659490ab01dfdb00e116baf35',
        '7e7555fb09bf5bfbc8d337700a401b477a5697ca',
        '94542890b819fa16784c92d911b08e13bf3ed149',
        '30407ea7969464870ed5f70d9e8b7f5a89fe1688',
        '74d638978f5590925ea6a88ec57c71928efec166',
        '4333424bbdc1af485bc999994ab0d9594f0910be',
        '79e06166183e511764696155833e84e6fdbe8238',
        '0f809d34aa6d3bc87dc9ae4c5ee650e7c2bcf0fa',
        'dfc550842fbb3eaf4d4ab1ae234f76ec017762a5',
        '71673eae130ab52bb1606f5d13ecb9d4edb56503',
        '6d713a61738cb0e4438e5483e85c6eb8d556a6a7',
        'b25e50db034df2a5a88511ff315617cdd14aa953',
        '5e23a81e1a89a75823d418ca7618216f4ba2b2e9',
        'd580d727fe8fc3712621a48ab634dc5c60f74ad5',
        '362d09d1c64a54eae4212d1b87d0e8f7bd57c0f4',
        'f8c04dc8a399399b5bbb5de98f0e094e347745c0',
        '93c6ae43eeb7f69d746cfc86c494f4a342784961',
        '6a6c35b671f464a85dc842508f0294cb005bdaaa',
        '655c11e4edba22d8ca706777b1f99d9d6c6f8428',
        'cfdd6f2dcb6df705d645133fd4a27f838a4f60be',
        '98039b652aec3e72e2f78039288d33feb546f08f'
    ]

    for filename, drive_id, sha1_hex in zip(filenames, drive_ids, sha):
        print('Deal with file: ' + filename)
        save_path = os.path.join(global_data_dir, filename)

        if os.path.exists(save_path):
            with open(save_path) as f:
                file_content = f.read()
            if hashlib.sha1(file_content).hexdigest() == sha1_hex:
                print('[*] {} already exists'.format(save_path))
                continue
            else:
                os.remove(save_path)
        
        download_file_from_google_drive(drive_id, save_path)

        # zip_dir = ''
        # with zipfile.ZipFile(save_path) as zf:
        #     zip_dir = zf.namelist()[0]
        #     zf.extractall(global_data_dir)
        # os.remove(save_path)