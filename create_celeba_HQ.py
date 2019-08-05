# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import glob
import argparse
import threading
import Queue
import traceback
import numpy as np
import scipy.ndimage
import PIL.Image


class ExceptionInfo(object):
    def __init__(self):
        self.type, self.value = sys.exc_info()[:2]
        self.traceback = traceback.format_exc()


# ----------------------------------------------------------------------------


class WorkerThread(threading.Thread):
    def __init__(self, task_queue):
        threading.Thread.__init__(self)
        self.task_queue = task_queue

    def run(self):
        while True:
            func, args, result_queue = self.task_queue.get()
            if func is None:
                break
            try:
                result = func(*args)
            except:
                result = ExceptionInfo()
            result_queue.put((result, args))


#----------------------------------------------------------------------------


class ThreadPool(object):
    def __init__(self, num_threads):
        assert num_threads >= 1
        self.task_queue = Queue.Queue()
        self.result_queues = dict()
        self.num_threads = num_threads
        for idx in xrange(self.num_threads):
            thread = WorkerThread(self.task_queue)
            thread.daemon = True
            thread.start()

    def add_task(self, func, args=()):
        assert hasattr(func, '__call__')  # must be a function
        if func not in self.result_queues:
            self.result_queues[func] = Queue.Queue()
        self.task_queue.put((func, args, self.result_queues[func]))

    def get_result(self, func,
                   verbose_exceptions=True):  # returns (result, args)
        result, args = self.result_queues[func].get()
        if isinstance(result, ExceptionInfo):
            if verbose_exceptions:
                print '\n\nWorker thread caught an exception:\n' + result.traceback + '\n',
            raise result.type, result.value
        return result, args

    def finish(self):
        for idx in xrange(self.num_threads):
            self.task_queue.put((None, (), None))

    def __enter__(self):  # for 'with' statement
        return self

    def __exit__(self, *excinfo):
        self.finish()

    def process_items_concurrently(self,
                                   item_iterator,
                                   process_func=lambda x: x,
                                   pre_func=lambda x: x,
                                   post_func=lambda x: x,
                                   max_items_in_flight=None):
        if max_items_in_flight is None:
            max_items_in_flight = self.num_threads * 4
        assert max_items_in_flight >= 1
        results = []
        retire_idx = [0]

        def task_func(prepared, idx):
            return process_func(prepared)

        def retire_result():
            processed, (prepared, idx) = self.get_result(task_func)
            results[idx] = processed
            while retire_idx[0] < len(
                    results) and results[retire_idx[0]] is not None:
                yield post_func(results[retire_idx[0]])
                results[retire_idx[0]] = None
                retire_idx[0] += 1

        for idx, item in enumerate(item_iterator):
            prepared = pre_func(item)
            results.append(None)
            self.add_task(func=task_func, args=(prepared, idx))
            while retire_idx[0] < idx - max_items_in_flight + 2:
                for res in retire_result():
                    yield res
        while retire_idx[0] < len(results):
            for res in retire_result():
                yield res


# ----------------------------------------------------------------------------


def create_celeba_hq(celeba_dir,
                     delta_dir,
                     output_dir,
                     num_threads=4,
                     num_tasks=100):
    print 'Loading CelebA data from %s' % celeba_dir
    glob_pattern = os.path.join(celeba_dir, 'img_celeba', '*.jpg')
    glob_expected = 202599
    if len(glob.glob(glob_pattern)) != glob_expected:
        print 'Error: Expected to find %d images in %s' % (glob_expected,
                                                           glob_pattern)
        return
    with open(
            os.path.join(celeba_dir, 'Anno', 'list_landmarks_celeba.txt'),
            'rt') as file:
        landmarks = [[float(value) for value in line.split()[1:]]
                     for line in file.readlines()[2:]]
        for i in range(len(landmarks)):
            if (len(landmarks[i]) != 10):
                landmarks[i] = [0] * 10
            a = np.reshape(landmarks[i], [5, 2])
            landmarks[i] = a
        landmarks = np.array(landmarks)
        print(landmarks.shape)

    print 'Loading CelebA-HQ deltas from %s' % delta_dir
    import hashlib
    import bz2
    import zipfile
    import base64
    import cryptography.hazmat.primitives.hashes
    import cryptography.hazmat.backends
    import cryptography.hazmat.primitives.kdf.pbkdf2
    import cryptography.fernet
    glob_pattern = os.path.join(delta_dir, 'delta*.zip')
    glob_expected = 30
    if len(glob.glob(glob_pattern)) != glob_expected:
        print 'Error: Expected to find %d zips in %s' % (glob_expected,
                                                         glob_pattern)
        return
    with open(os.path.join(delta_dir, 'image_list.txt'), 'rt') as file:
        lines = [line.split() for line in file]
        fields = dict()
        for idx, field in enumerate(lines[0]):
            type = int if field.endswith('idx') else str
            fields[field] = [type(line[idx]) for line in lines[1:]]

    def rot90(v):
        return np.array([-v[1], v[0]])

    def process_func(idx):
        # Load original image.
        orig_idx = fields['orig_idx'][idx]
        orig_file = fields['orig_file'][idx]
        orig_path = os.path.join(celeba_dir, 'img_celeba', orig_file)
        img = PIL.Image.open(orig_path)

        # Choose oriented crop rectangle.
        lm = landmarks[orig_idx]
        eye_avg = (lm[0] + lm[1]) * 0.5 + 0.5
        mouth_avg = (lm[3] + lm[4]) * 0.5 + 0.5
        eye_to_eye = lm[1] - lm[0]
        eye_to_mouth = mouth_avg - eye_avg
        x = eye_to_eye - rot90(eye_to_mouth)
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = rot90(x)
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        zoom = 1024 / (np.hypot(*x) * 2)

        # Shrink.
        shrink = int(np.floor(0.5 / zoom))
        if shrink > 1:
            size = (int(np.round(float(img.size[0]) / shrink)),
                    int(np.round(float(img.size[1]) / shrink)))
            img = img.resize(size, PIL.Image.ANTIALIAS)
            quad /= shrink
            zoom *= shrink

        # Crop.
        border = max(int(np.round(1024 * 0.1 / zoom)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
                int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
                min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Simulate super-resolution.
        superres = int(np.exp2(np.ceil(np.log2(zoom))))
        if superres > 1:
            img = img.resize((img.size[0] * superres, img.size[1] * superres),
                             PIL.Image.ANTIALIAS)
            quad *= superres
            zoom /= superres

        # Pad.
        pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
               int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0),
               max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))
        if max(pad) > border - 4:
            pad = np.maximum(pad, int(np.round(1024 * 0.3 / zoom)))
            img = np.pad(
                np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                'reflect')
            h, w, _ = img.shape
            y, x, _ = np.mgrid[:h, :w, :1]
            mask = 1.0 - np.minimum(
                np.minimum(np.float32(x) / pad[0],
                           np.float32(y) / pad[1]),
                np.minimum(
                    np.float32(w - 1 - x) / pad[2],
                    np.float32(h - 1 - y) / pad[3]))
            blur = 1024 * 0.02 / zoom
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) -
                    img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(
                mask, 0.0, 1.0)
            img = PIL.Image.fromarray(
                np.uint8(np.clip(np.round(img), 0, 255)), 'RGB')
            quad += pad[0:2]

        # Transform.
        img = img.transform((4096, 4096), PIL.Image.QUAD,
                            (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        img = img.resize((1024, 1024), PIL.Image.ANTIALIAS)
        img = np.asarray(img).transpose(2, 0, 1)

        # Load delta image and original JPG.
        with zipfile.ZipFile(
                os.path.join(delta_dir, 'deltas%05d.zip' % (idx - idx % 1000)),
                'r') as zip:
            delta_bytes = zip.read('delta%05d.dat' % idx)
        with open(orig_path, 'rb') as file:
            orig_bytes = file.read()

        # Decrypt delta image, using original JPG data as decryption key.
        algorithm = cryptography.hazmat.primitives.hashes.SHA256()
        backend = cryptography.hazmat.backends.default_backend()
        kdf = cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC(
            algorithm=algorithm,
            length=32,
            salt=orig_file,
            iterations=100000,
            backend=backend)
        key = base64.urlsafe_b64encode(kdf.derive(orig_bytes))
        delta = np.frombuffer(
            bz2.decompress(
                cryptography.fernet.Fernet(key).decrypt(delta_bytes)),
            dtype=np.uint8).reshape(3, 1024, 1024)

        # Apply delta image.
        img = img + delta
        img = np.asarray(img).transpose(1, 2, 0)
        img = PIL.Image.fromarray(img, mode='RGB')
        img512 = img.resize((512, 512), PIL.Image.ANTIALIAS)
        img256 = img.resize((256, 256), PIL.Image.ANTIALIAS)
        img128 = img.resize((128, 128), PIL.Image.ANTIALIAS)
        img64 = img.resize((64, 64), PIL.Image.ANTIALIAS)
        return orig_file, img64, img128, img256, img512, img

    img64dir = os.path.join(output_dir, 'celeba-hq', 'celeba-64')
    img128dir = os.path.join(output_dir, 'celeba-hq', 'celeba-128')
    img256dir = os.path.join(output_dir, 'celeba-hq', 'celeba-256')
    img512dir = os.path.join(output_dir, 'celeba-hq', 'celeba-512')
    img1024dir = os.path.join(output_dir, 'celeba-hq', 'celeba-1024')
    output_dir_size = [img64dir, img128dir, img256dir, img512dir, img1024dir]
    for paths in output_dir_size:
        if not os.path.exists(paths):
            os.makedirs(paths)

    # Save all generated images.
    with ThreadPool(num_threads) as pool:
        for orig_fn, aimg64, aimg128, aimg256, aimg512, aimg1024 in pool.process_items_concurrently(
                fields['idx'],
                process_func=process_func,
                max_items_in_flight=num_tasks):
            aimg64.save(img64dir + os.sep + str(orig_fn))
            aimg128.save(img128dir + os.sep + str(orig_fn))
            aimg256.save(img256dir + os.sep + str(orig_fn))
            aimg512.save(img512dir + os.sep + str(orig_fn))
            aimg1024.save(img1024dir + os.sep + str(orig_fn))
            print('Generated image: ' + str(orig_fn))


# ----------------------------------------------------------------------------


def get_parser():
    p = argparse.ArgumentParser(
        description='Create HDF5 dataset for CelebA-HQ.')

    p.add_argument(
        '--h5_filename',
        type=str,
        default='celeba-hq-1024x1024.h5',
        help='HDF5 file to create')
    p.add_argument(
        '--celeba_dir', type=str, help='Directory to read CelebA data from')
    p.add_argument(
        '--delta_dir',
        type=str,
        help='Directory to read CelebA-HQ deltas from')
    p.add_argument('--output_dir', type=str, help='Directory to write images')
    p.add_argument(
        '--num_threads',
        help='Number of concurrent threads (default: 4)',
        type=int,
        default=None)
    p.add_argument(
        '--num_tasks',
        help='Number of concurrent processing tasks (default: 100)',
        type=int,
        default=100)

    return p


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.num_threads is None:
        args.num_threads = os.sysconf('SC_NPROCESSORS_ONLN')
    create_celeba_hq(
        args.celeba_dir,
        args.delta_dir,
        args.output_dir,
        num_threads=args.num_threads,
        num_tasks=args.num_tasks)

# ----------------------------------------------------------------------------
