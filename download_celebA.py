import tarfile
import zipfile
import gzip
import os
import hashlib
import sys
from glob import glob

if sys.version_info[0] > 2:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
from subprocess import Popen
import argparse
from download_celebA_HQ import download_file_from_google_drive

parser = argparse.ArgumentParser(description='Download celebA helper')
parser.add_argument('path', type=str)


def require_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def checksum(filename, method='sha1'):
    data = open(filename, 'rb').read()
    if method == 'sha1':
        return hashlib.sha1(data).hexdigest()
    elif method == 'md5':
        return hashlib.md5(data).hexdigest()
    else:
        raise ValueError('Invalid method: %s' % method)
    return None


def download(url, target_dir, filename=None):
    require_dir(target_dir)
    if filename is None:
        filename = url_filename(url)
    filepath = os.path.join(target_dir, filename)
    urlretrieve(url, filepath)
    return filepath


def url_filename(url):
    return url.split('/')[-1].split('#')[0].split('?')[0]


def archive_extract(filepath, target_dir):
    target_dir = os.path.abspath(target_dir)
    if tarfile.is_tarfile(filepath):
        with tarfile.open(filepath, 'r') as tarf:
            # Check that no files get extracted outside target_dir
            for name in tarf.getnames():
                abs_path = os.path.abspath(os.path.join(target_dir, name))
                if not abs_path.startswith(target_dir):
                    raise RuntimeError('Archive tries to extract files '
                                       'outside target_dir.')
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tarf, target_dir)
    elif zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, 'r') as zipf:
            zipf.extractall(target_dir)
    elif filepath[-3:].lower() == '.gz':
        with gzip.open(filepath, 'rb') as gzipf:
            with open(filepath[:-3], 'wb') as outf:
                outf.write(gzipf.read())
    elif '.7z' in filepath:
        if os.name != 'posix':
            raise NotImplementedError('Only Linux and Mac OS X support .7z '
                                      'compression.')
        print('Using 7z!!!')
        cmd = '7z x {} -o{}'.format(filepath, target_dir)
        retval = Popen(cmd, shell=True).wait()
        if retval != 0:
            raise RuntimeError(
                'Archive file extraction failed for {}.'.format(filepath))
    elif filepath[-2:].lower() == '.z':
        if os.name != 'posix':
            raise NotImplementedError('Only Linux and Mac OS X support .Z '
                                      'compression.')
        cmd = 'gzip -d {}'.format(filepath)
        retval = Popen(cmd, shell=True).wait()
        if retval != 0:
            raise RuntimeError(
                'Archive file extraction failed for {}.'.format(filepath))
    else:
        raise ValueError('{} is not a supported archive file.'.format(filepath))


def download_and_check(drive_data, path):
    save_paths = list()
    n_files = len(drive_data["filenames"])
    for i in range(n_files):
        drive_id = drive_data["drive_ids"][i]
        filename = drive_data["filenames"][i]
        save_path = os.path.join(path, filename)
        require_dir(os.path.dirname(save_path))
        print('Downloading {} to {}'.format(filename, save_path))
        sha1 = drive_data["sha1"][i]
        
        if os.path.exists(save_path) and sha1 == checksum(save_path, 'sha1'):
            print('[*] {} already exists'.format(save_path))
            continue

        download_file_from_google_drive(drive_id, save_path)
        print('Done!')
        
        print('Check SHA1 {}'.format(save_path))
        if sha1 != checksum(save_path, 'sha1'):
            raise RuntimeError('Checksum mismatch for %s.' % save_path)
        save_paths.append(save_path)
    return save_paths


def download_celabA(dataset_dir):
    _IMGS_DRIVE = dict(
            filenames = [
                'img_celeba.7z.001', 'img_celeba.7z.002', 'img_celeba.7z.003',
                'img_celeba.7z.004', 'img_celeba.7z.005', 'img_celeba.7z.006',
                'img_celeba.7z.007', 'img_celeba.7z.008', 'img_celeba.7z.009',
                'img_celeba.7z.010', 'img_celeba.7z.011', 'img_celeba.7z.012',
                'img_celeba.7z.013', 'img_celeba.7z.014'
                ],
            drive_ids = [
                '0B7EVK8r0v71pQy1YUGtHeUM2dUE', '0B7EVK8r0v71peFphOHpxODd5SjQ',
                '0B7EVK8r0v71pMk5FeXRlOXcxVVU', '0B7EVK8r0v71peXc4WldxZGFUbk0',
                '0B7EVK8r0v71pMktaV1hjZUJhLWM', '0B7EVK8r0v71pbWFfbGRDOVZxOUU',
                '0B7EVK8r0v71pQlZrOENSOUhkQ3c', '0B7EVK8r0v71pLVltX2F6dzVwT0E',
                '0B7EVK8r0v71pVlg5SmtLa1ZiU0k', '0B7EVK8r0v71pa09rcFF4THRmSFU',
                '0B7EVK8r0v71pNU9BZVBEMF9KN28', '0B7EVK8r0v71pTVd3R2NpQ0FHaGM',
                '0B7EVK8r0v71paXBad2lfSzlzSlk', '0B7EVK8r0v71pcTFwT1VFZzkzZk0'
                ],
            sha1 = [
                '8591a74c4b5bc8d31f975c869807cbff8ccd1541',
                'ecc1e0e0c6fd19959ba045d4b1dc0cd621541a2f',
                'cf6d8ba274401fbfb471199dae2786184948a74c',
                '2a08f012cfbce90bebf3f4422b52232c4bef98d5',
                'bcdb8fad2bae91b610e61bde643e5e442d36450d',
                'da36d7bdc8b0da1568662705c6f8c6b85f0e247e',
                '27977d05b152bbd243785b25c159c62689f01ad1',
                'dc266301ba41c32b33de06bde863995b99276841',
                'c59ac24d21151437f5bb851745c5369bbf22cb6c',
                '858dbb3befc78a664ac51115d86d4199712038a3',
                'feadf47f96e0e5000c21c9959bd497b8247c90bb',
                'd54c4c02a1789d7ade90cc42a0525680f926f6ca',
                'ab337954da2e7940fcf18b2b957e03601891f843',
                'cb6c97189beb560c7d777960cfd511505e8b8af0'
                ]
            )

    _ATTRIBUTES_DRIVE = dict(
            filenames = [
                'Anno/list_landmarks_celeba.txt',
                'Anno/list_landmarks_align_celeba.txt',
                'Anno/list_bbox_celeba.txt',
                'Anno/list_attr_celeba.txt',
                'Anno/identity_CelebA.txt'
                ],
            drive_ids = [
                '0B7EVK8r0v71pTzJIdlJWdHczRlU',
                '0B7EVK8r0v71pd0FJY3Blby1HUTQ',
                '0B7EVK8r0v71pbThiMVRxWXZ4dU0',
                '0B7EVK8r0v71pblRyaVFSWGxPY0U',
                '1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS'
                ],
            sha1 = [
                'ea255cd0ffe98ca88bff23767f7a5ece7710db57',
                'd23d12ca9cb01ef2fe9abc44ef48d0e12198785c',
                '173a25764dafa45b1d8383adf0d7fa10a3ab2476',
                '225788ff6c9d0b96dc21144147456e0388195617',
                'ed25ac86acb7fac1c6baea876b06adea31f68277'
                ]
            )

    download_and_check(_ATTRIBUTES_DRIVE, dataset_dir)
    download_and_check(_IMGS_DRIVE, dataset_dir)

    return True

if __name__ == '__main__':
    args = parser.parse_args()
    dirpath = args.path
    dataset_dir = os.path.join(dirpath, 'celebA')
    download_celabA(dataset_dir)
