import codecs
import errno
import gzip
import imageio
import math
import numbers
import numpy as np
import os
import random
import torch
import torchvision
from PIL import Image, ImageOps
from six.moves import urllib
from torch.utils import data
from torchvision import transforms
from typing import Union


class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz'
        # 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        # 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        # 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        # 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    multi_training_file = 'multi_training.pt'
    multi_test_file = 'multi_test.pt'

    def __init__(self, root: str,
                 train: bool = True,
                 transform: bool = None,
                 target_transform: str = None,
                 download: bool = False,
                 multi: bool = False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.multi = multi

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if not self._check_multi_exists():
            raise RuntimeError('Multi Task extension not found.' +
                               ' You can use download=True to download it')

        if multi:
            if self.train:
                self.train_data, self.train_labels_l, self.train_labels_r = torch.load(
                    os.path.join(self.root, self.processed_folder, self.multi_training_file))
            else:
                self.test_data, self.test_labels_l, self.test_labels_r = torch.load(
                    os.path.join(self.root, self.processed_folder, self.multi_test_file))
        else:
            if self.train:
                self.train_data, self.train_labels = torch.load(
                    os.path.join(self.root, self.processed_folder, self.training_file))
            else:
                self.test_data, self.test_labels = torch.load(
                    os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index: int) -> Union[torch.Tensor, int]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.multi:
            if self.train:
                img, target_l, target_r = self.train_data[index], self.train_labels_l[index], self.train_labels_r[index]
            else:
                img, target_l, target_r = self.test_data[index], self.test_labels_l[index], self.test_labels_r[index]
        else:
            if self.train:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy().astype(np.uint8), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.multi:
            return img, [target_l, target_r]
        else:
            return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def _check_multi_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists() and self._check_multi_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')
        mnist_ims, multi_mnist_ims, extension = read_image_file(
            os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte'))
        mnist_labels, multi_mnist_labels_l, multi_mnist_labels_r = read_label_file(
            os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'), extension)

        tmnist_ims, tmulti_mnist_ims, textension = read_image_file(
            os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte'))
        tmnist_labels, tmulti_mnist_labels_l, tmulti_mnist_labels_r = read_label_file(
            os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'), textension)

        mnist_training_set = (mnist_ims, mnist_labels)
        multi_mnist_training_set = (multi_mnist_ims, multi_mnist_labels_l, multi_mnist_labels_r)

        mnist_test_set = (tmnist_ims, tmnist_labels)
        multi_mnist_test_set = (tmulti_mnist_ims, tmulti_mnist_labels_l, tmulti_mnist_labels_r)

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(mnist_training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(mnist_test_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.multi_training_file), 'wb') as f:
            torch.save(multi_mnist_training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.multi_test_file), 'wb') as f:
            torch.save(multi_mnist_test_set, f)
        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path: str,
                    extension: str) -> torch.Tensor:
    """
    Args:
        path: path to label file
        extension: format of file

    Returns:

    """
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        multi_labels_l = np.zeros((1 * length), dtype=np.long)
        multi_labels_r = np.zeros((1 * length), dtype=np.long)
        for im_id in range(length):
            for rim in range(1):
                multi_labels_l[1 * im_id + rim] = parsed[im_id]
                multi_labels_r[1 * im_id + rim] = parsed[extension[1 * im_id + rim]]
        return torch.from_numpy(parsed).view(length).long(), torch.from_numpy(multi_labels_l).view(
            length * 1).long(), torch.from_numpy(multi_labels_r).view(length * 1).long()


def read_image_file(path: str) -> torch.Tensor:
    """
    Args:
        path: path to image
    Returns:

    """
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        pv = parsed.reshape(length, num_rows, num_cols)
        multi_length = length * 1
        multi_data = np.zeros((1 * length, num_rows, num_cols))
        extension = np.zeros(1 * length, dtype=np.int32)
        for left in range(length):
            chosen_ones = np.random.permutation(length)[:1]
            extension[left * 1:(left + 1) * 1] = chosen_ones
            for j, right in enumerate(chosen_ones):
                lim = pv[left, :, :]
                rim = pv[right, :, :]
                new_im = np.zeros((36, 36))
                new_im[0:28, 0:28] = lim
                new_im[6:34, 6:34] = rim
                new_im[6:28, 6:28] = np.maximum(lim[6:28, 6:28], rim[0:22, 0:22])
                multi_data_im = np.array(Image.fromarray(new_im).resize((28, 28), resample=Image.NEAREST))
                # Inequal MNIST
                # rim =  np.array(Image.fromarray(rim).resize((14, 14), resample=Image.NEAREST))
                # new_im = np.zeros((28,28))
                # new_im[0:28,0:28] = lim
                # new_im[14:28,14:28] = rim
                # new_im[14:28,14:28] = np.maximum(lim[14:28,14:28], rim[0:14,0:14])
                # multi_data_im = np.array(Image.fromarray(new_im).resize((28, 28), resample=Image.NEAREST))
                multi_data[left * 1 + j, :, :] = multi_data_im

    return torch.from_numpy(parsed).view(length, num_rows, num_cols), torch.from_numpy(multi_data).view(length,
                                                                                                        num_rows,
                                                                                                        num_cols), extension


def global_transformer():
    """
    Transformations
    """
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])


class CustomDataset(data.Dataset):
    """
    Dataset for CIFAR10 experiment
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class CIFAR10Loader():
    """
    Loader for CIFAR10
    """

    def __init__(self, root, train=True):
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )

        dataset = torchvision.datasets.CIFAR10(root=root, train=train,
                                               download=True, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      drop_last=False)

    def get_loader(self, batch_size=128, N_WORKERS=4, shuffle=False, drop_last=False):
        images = []
        labels = []
        for batch_images, batch_labels in self.dataloader:
            for i in batch_images:
                images.append(i)
            for l in batch_labels:
                prom = torch.zeros(10)
                prom[l] = 1
                labels.append(prom)

        dataset = CustomDataset(images, labels)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 drop_last=drop_last,
                                                 num_workers=N_WORKERS)

        return dataloader


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


class Compose():
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask, ins, depth):
        img, mask, ins, depth = Image.fromarray(img, mode='RGB'), Image.fromarray(mask, mode='L'), Image.fromarray(ins,
                                                                                                                   mode='I'), Image.fromarray(
            depth, mode='F')
        assert img.size == mask.size
        assert img.size == ins.size
        assert img.size == depth.size

        for a in self.augmentations:
            img, mask, ins, depth = a(img, mask, ins, depth)

        return np.array(img), np.array(mask, dtype=np.uint8), np.array(ins, dtype=np.uint64), np.array(depth,
                                                                                                       dtype=np.float32)


class RandomCrop():
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, ins, depth):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            ins = ImageOps.expand(ins, border=self.padding, fill=0)
            depth = ImageOps.expand(depth, border=self.padding, fill=0)

        assert img.size == mask.size
        assert img.size == ins.size
        assert img.size == depth.size

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask, ins, depth
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST), ins.resize((tw, th),
                                                                                                          Image.NEAREST), depth.resize(
                (tw, th), Image.NEAREST)

        _sysrand = random.SystemRandom()
        x1 = _sysrand.randint(0, w - tw)
        y1 = _sysrand.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)), ins.crop(
            (x1, y1, x1 + tw, y1 + th)), depth.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop():
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip():
    def __call__(self, img, mask, ins, depth):
        _sysrand = random.SystemRandom()
        if _sysrand.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), ins.transpose(
                Image.FLIP_LEFT_RIGHT), depth.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask, ins, depth


class FreeScale():
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask, depth, ins=None):
        assert img.size == mask.size
        assert img.size == ins.size
        assert img.size == depth.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST), ins.resize(self.size,
                                                                                                        Image.NEAREST), depth.resize(
            self.size, Image.NEAREST)


class Scale():
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask, ins, depth):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask, ins, depth
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST), ins.resize((ow, oh),
                                                                                                          Image.NEAREST), depth.resuze(
                (ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST), ins.resize((ow, oh),
                                                                                                          Image.NEAREST), depth.reszie(
                (ow, oh), Image.NEAREST)


class RandomSizedCrop():
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        _sysrand = random.SystemRandom()
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]

            target_area = _sysrand.uniform(0.45, 1.0) * area
            aspect_ratio = _sysrand.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if _sysrand.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = _sysrand.randint(0, img.size[0] - w)
                y1 = _sysrand.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                       Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate():
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask, ins, depth):
        _sysrand = random.SystemRandom()
        rotate_degree = _sysrand.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST), ins.rotate(
            rotate_degree, Image.NEAREST), depth.rotate(rotate_degree, Image.NEAREST)


class RandomSized():
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size
        _sysrand = random.SystemRandom()

        w = int(_sysrand.uniform(0.5, 2) * img.size[0])
        h = int(_sysrand.uniform(0.5, 2) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return self.crop(*self.scale(img, mask))


class CITYSCAPES(data.Dataset):
    """cityscapesLoader
    https://www.cityscapes-dataset.com
    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]]

    label_colours = dict(zip(range(19), colors))

    def __init__(self, root, split=["train"], is_transform=True,
                 img_size=(512, 1024), augmentations=None):
        """__init__
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.split_text = '+'.join(split)
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 19
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([123.675, 116.28, 103.53])
        # self.mean = np.array([73.15835921, 82.90891754, 72.39239876])
        self.files = {}

        self.files[self.split_text] = []
        for _split in self.split:
            self.images_base = os.path.join(self.root, 'leftImg8bit', _split)
            self.annotations_base = os.path.join(self.root, 'gtFine', _split)
            self.files[self.split_text] = recursive_glob(rootdir=self.images_base, suffix='.png')
            self.depth_base = os.path.join(self.root, 'disparity', _split)

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.no_instances = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if len(self.files[self.split_text]) < 2:
            raise Exception("No files for split=[%s] found in %s" % (self.split_text, self.images_base))

        print("Found %d %s images" % (len(self.files[self.split_text]), self.split_text))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split_text])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split_text][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')
        instance_path = os.path.join(self.annotations_base,
                                     img_path.split(os.sep)[-2],
                                     os.path.basename(img_path)[:-15] + 'gtFine_instanceIds.png')
        depth_path = os.path.join(self.depth_base,
                                  img_path.split(os.sep)[-2],
                                  os.path.basename(img_path)[:-15] + 'disparity.png')
        img = imageio.imread(img_path)
        lbl = imageio.imread(lbl_path)
        ins = imageio.imread(instance_path)
        depth = np.array(imageio.imread(depth_path), dtype=np.float32)

        if self.augmentations is not None:
            img, lbl, ins, depth = self.augmentations(np.array(img, dtype=np.uint8), np.array(lbl, dtype=np.uint8),
                                                      np.array(ins, dtype=np.int32), np.array(depth, dtype=np.float32))

        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        ins_y, ins_x = self.encode_instancemap(lbl, ins)
        # Zero-Mean, Std-Dev depth map

        if self.is_transform:
            img, lbl, ins_gt, depth = self.transform(img, lbl, ins_y, ins_x, depth)

        return img, (lbl, ins_gt, depth)

    def transform(self, img, lbl, ins_y, ins_x, depth):
        """transform
        :param img:
        :param lbl:
        """
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        # Maybe use cv2 package
        img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((self.img_size[0], self.img_size[1])))
        # img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = np.array(
            Image.fromarray((lbl).astype(np.uint8)).resize((int(self.img_size[0] / 8), int(self.img_size[1] / 8))))
        # lbl = m.imresize(lbl, (int(self.img_size[0]/8), int(self.img_size[1]/8)), 'nearest', mode='F') # TODO(ozan) /8 is quite hacky
        lbl = lbl.astype(int)

        ins_y = ins_y.astypef(float)
        ins_y = np.array(
            Image.fromarray((ins_y).astype(np.uint8)).resize((int(self.img_size[0] / 8), int(self.img_size[1] / 8))))
        # ins_y = m.imresize(ins_y, (int(self.img_size[0]/8), int(self.img_size[1]/8)), 'nearest', mode='F')

        ins_x = ins_x.astype(float)
        ins_x = np.array(
            Image.fromarray((ins_x).astype(np.uint8)).resize((int(self.img_size[0] / 8), int(self.img_size[1] / 8))))
        # ins_x = m.imresize(ins_x, (int(self.img_size[0]/8), int(self.img_size[1]/8)), 'nearest', mode='F')

        depth = np.array(
            Image.fromarray((depth).astype(np.uint8)).resize((int(self.img_size[0] / 8), int(self.img_size[1] / 8))))
        # depth = m.imresize(depth, (int(self.img_size[0]/8), int(self.img_size[1]/8)), 'nearest', mode='F')
        depth = np.expand_dims(depth, axis=0)
        # if not np.all(classes == np.unique(lbl)):
        #    print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print('after det', classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        ins = np.stack((ins_y, ins_x))
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        ins = torch.from_numpy(ins).float()
        depth = torch.from_numpy(depth).float()
        return img, lbl, ins, depth

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def encode_instancemap(self, mask, ins):
        ins[mask == self.ignore_index] = self.ignore_index
        for _no_instance in self.no_instances:
            ins[ins == _no_instance] = self.ignore_index
        ins[ins == 0] = self.ignore_index

        instance_ids = np.unique(ins)
        sh = ins.shape
        ymap, xmap = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]), indexing='ij')

        out_ymap, out_xmap = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]), indexing='ij')
        out_ymap = np.ones(ymap.shape) * self.ignore_index
        out_xmap = np.ones(xmap.shape) * self.ignore_index

        for instance_id in instance_ids:
            if instance_id == self.ignore_index:
                continue
            instance_indicator = (ins == instance_id)
            coordinate_y, coordinate_x = np.mean(ymap[instance_indicator]), np.mean(xmap[instance_indicator])
            out_ymap[instance_indicator] = ymap[instance_indicator] - coordinate_y
            out_xmap[instance_indicator] = xmap[instance_indicator] - coordinate_x

        return out_ymap, out_xmap
