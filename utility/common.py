#! /usr/bin/env python
# coding: utf-8

import os
import json
import glob
import yaml
import time
import pickle
import logging
import datetime
import functools
import collections
import multiprocessing

import tqdm


_global_lock = multiprocessing.Lock()


def print_block():
    print("===========================================================")

def collect_images_by_index(image_folder_path, cam_id):
    images_prefix =f"{image_folder_path}/*_{cam_id}.jpg"
    images_paths = glob.glob(images_prefix)
    return images_paths

def create_ouput_folder(prefix="sample"):
    """
    check and create asending order new directory with prefix
    """
    record_path = os.path.join(os.path.curdir, "data", "record")
    os.makedirs(record_path,exist_ok=True)
    n = 0
    while True:
        sample_path = os.path.join(record_path,f"{prefix}_{n}")
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
            return sample_path
        else:
            n +=1

def create_asending_folder(path, prefix="sample"):
    """
    check and create asending order new directory with prefix
    """

    record_path = os.path.join(os.path.curdir, path)
    os.makedirs(record_path,exist_ok=True)
    n = 0
    while True:
        sample_path = os.path.join(record_path,f"{prefix}_{n}")
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
            return sample_path
        else:
            n +=1



def prepare_dir(path):
    """Make dir if necessary."""

    dirname = os.path.dirname(path)
    # 当前目录直接返回
    if not dirname: return

    with _global_lock:
        os.makedirs(dirname, exist_ok=True)


def normlize_path(path):
    """normalize path to standard format."""

    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    path = os.path.normpath(path)
    return path


def is_valid_file(path):
    """Return True if path is valid."""

    if not path: return False
    if not os.path.exists(path): return False
    return os.path.getsize(path) > 0


def read_yaml_file(path, *, check=True):
    valid = is_valid_file(path)
    assert (not check) or valid, f"Failed to read file: {path}"
    if not valid: return None
    with open(path) as f:
        return yaml.safe_load(f)


def read_binary_file(path, *, check=True):
    """Read raw data from file."""

    valid = is_valid_file(path)
    assert (not check) or valid, f"Failed to read file: {path}"
    if not valid: return None
    with open(path, "rb") as srcfile:
        return srcfile.read()


def read_pickle_file(path, *, check=True):
    """Read file in pickle format."""

    valid = is_valid_file(path)
    assert (not check) or valid, f"Failed to read file: {path}"
    if not valid: return None
    with open(path, "rb") as srcfile:
        return pickle.load(srcfile)


def read_json_file(path, *, check=True):
    """Read file in json format."""

    # valid = is_valid_file(path)
    # assert (not check) or valid, f"Failed to read file: {path}"
    # if not valid: return None
    with open(path, "rb") as srcfile:
        return json.load(srcfile)


def read_list_file(path, sep=None, *, check=True):
    """Read file as a list of strings."""

    valid = is_valid_file(path)
    assert (not check) or valid, f"Failed to read file: {path}"
    if not valid: return None
    with open(path, "r", encoding="utf-8") as srcfile:
        lines = [l.strip() for l in srcfile]
        # 去掉空行和注释行(以'#'开头的行)
        lines = [l for l in lines if l and (not l.startswith("#"))]
    if isinstance(sep, str):
        lines = [l.split(sep) for l in lines]
        lines = [tuple(filter(None, l)) for l in lines]
    return lines


def read_list_field(path, field=0, sep=" ", *, check=True):
    """Read file and extract field."""

    lines = read_list_file(path, sep, check=check)
    if not lines: return None
    if isinstance(field, int):
        return [n[field] for n in lines]
    return [[n[i] for i in field] for n in lines]


def read_map_file(path, vtype=str, *, check=True):
    """Read file as a dictionay."""

    lines = read_list_file(path, " ", check=check)
    if not lines: return None
    return {k: vtype(v) for k, v in lines}


def write_binary_file(data, path):
    """Write data to pickle file."""

    prepare_dir(path)
    if isinstance(data, str):
        data = data.encode("utf-8")
    with open(path, "wb") as dstfile:
        dstfile.write(data)


def write_pickle_file(data, path):
    """Write data to pickle file."""

    prepare_dir(path)
    with open(path, "wb") as dstfile:
        pickle.dump(data, dstfile, protocol=pickle.HIGHEST_PROTOCOL)


def write_json_file(data, path):
    """Write data to json file."""

    prepare_dir(path)
    with open(path, "w", encoding="utf-8") as dstfile:
        json.dump(data, dstfile, indent=2, ensure_ascii=False)
        dstfile.write("\n")


def write_list_file(data, path, sep=" "):
    """Write list to txt file."""

    prepare_dir(path)
    with open(path, "w", encoding="utf-8") as dstfile:
        for line in data:
            if isinstance(line, (tuple, list)):
                line = sep.join([str(item) for item in line])
            dstfile.write(line + "\n")


def traverse_directory(root, stopf=None, targetf=None):
    """Traverse directory.

    Args:
        targetf: function with one argument. Check whether a target path
            can be produced from the current path or not, if it is the
            case, return target path, otherwise return None. When function
            `targetf` returns a path, yield it.
        stopf: function with one argument. Check whether to enter current
            directory or not. If current path is a directory and this
            function returns False, enter current path.
    """

    # 默认状况下, 相当于listdir
    stopf = stopf or (lambda path: True)
    targetf = targetf or (lambda path: path if stopf(path) else None)
    if not os.path.exists(root): return
    for name in os.listdir(root):
        path = os.path.join(root, name)
        target = targetf(path)
        if target: yield target
        if os.path.isdir(path) and (not stopf(path)):
            yield from traverse_directory(path, stopf, targetf)


def initialize_logger(name=None, file=None, *, display=True):
    """Configurate logger based on settings.

    Args:
        name: str. Logger name, 'None' means root logger.
        file: str. Path of file to log to, 'None' or empty string to disable
            logging to file.
        display: bool. Indicate whether the logger logs to console or not.

    Returns:
        The required logger instance.
    """

    logger = logging.getLogger(name)
    if logger.handlers: return logger

    # 局部的logger是全局logger的child, 这里防止局部的log扩散到全局log中
    logger.propagate = False
    logger.setLevel(logging.INFO)

    # 这里行号设置为3个字符宽度, 因为我自己写的程序一个文件很少超过1000行
    fmt = "%(asctime)s %(filename)s:%(lineno)03d] %(levelname)s: %(message)s"
    fmt = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    if file:
        prepare_dir(file)
        handler = logging.FileHandler(file)
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    if display:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

    return logger


def get_global_logger(logging_root):
    date = str(datetime.date.today())
    logging_file = os.path.join(logging_root, date, "log_global.txt")
    prepare_dir(logging_file)
    return initialize_logger(None, logging_file, display=True)


def get_local_logger(name, logging_root):
    assert name != "global"
    date = str(datetime.date.today())
    logging_file = os.path.join(logging_root, date, f"log_{name}.txt")
    prepare_dir(logging_file)
    return initialize_logger(name, logging_file, display=False)


def get_progress_tracker(iterable=None, total=None):
    return tqdm.tqdm(
        iterable=iterable,
        total=total,
        mininterval=5,
        ascii=True,
        unit=" samples",
    )


def log_and_die(message, logger=None):
    logger = logger or logging.getLogger()
    logger.fatal(message)
    raise Exception(message)


def pass_or_die(assertion, message, logger=None):
    if not assertion: log_and_die(message, logger)


def format_time_interval(value, unit="s"):
    if unit == "ms" and value > 1000:
        unit, value = "s", value / 1000
    if unit == "s" and value > 60:
        unit, value = "m", value / 60
    if unit == "m" and value > 60:
        unit, value = "h", value / 60
    if unit == "h" and value > 24:
        unit, value = "d", value / 24
    return f"{value:.2f}{unit}"


def merge_overlap(groups, overlapf=None, mergef=None):
    # 默认情况下, groups中的元素是set类型
    overlapf = overlapf or (lambda x, y: x & y)
    mergef = mergef or (lambda x, y: x | y)
    if len(groups) < 2: return groups

    def _merge_one(anchor, samples):
        """将`samples`中的元素merge到`anchor`中."""
        remains = []
        for s in samples:
            if overlapf(anchor, s):
                anchor = mergef(anchor, s)
            else:
                remains.append(s)
        if len(remains) == len(samples):
            return anchor, remains
        return _merge_one(anchor, remains)

    anchor, remains = _merge_one(groups[0], groups[1:])
    return [anchor] + merge_overlap(remains, overlapf, mergef)


# 因为要在其他项目中使用, 这里不用默认值
def get_project_root(start):
    root = normlize_path(start)
    home = normlize_path("~")
    if not root.startswith(home): return None
    check_files = (".git", ".gitignore", ".neoignore", ".pylintrc")
    while not os.path.samefile(root, home):
        for name in check_files:
            path = os.path.join(root, name)
            if os.path.exists(path): return root
        root = os.path.dirname(root)
    return None


def group_by(samples, *, field=None, attr=None, keyf=None):
    # field用于tuple, list和dict, attr用于自定义class
    if field is not None:
        keyf = lambda x: x[field]
    elif attr is not None:
        keyf = lambda x: getattr(x, attr)

    assert keyf is not None
    groups = collections.defaultdict(list)
    for sample in samples:
        groups[keyf(sample)].append(sample)
    return groups


def group_by_key(samples, keys, *, keyf=None):
    keyf = keyf or (lambda x: x)
    groups = collections.defaultdict(list)
    for sample, key in zip(samples, keys):
        groups[keyf(key)].append(sample)
    return groups


# method取值: ["interlace", "adjacent"]
def group_by_index(samples, batch_size, *, method="adjacent"):
    if len(samples) == 0: return []
    count = (len(samples) + batch_size - 1) // batch_size
    groups = [[] for i in range(count)]
    if method == "adjacent":
        for i, sample in enumerate(samples):
            groups[i // batch_size].append(sample)
    elif method == "interlace":
        for i, sample in enumerate(samples):
            groups[i % batch_size].append(sample)
    return groups


def timestamp_to_date(timestamp):
    """unix timestamp (ms) to isofromat date."""

    timestamp = int(timestamp) // 1000
    return datetime.date.fromtimestamp(timestamp).isoformat()


def timestamp_to_datetime(timestamp, fmt="%Y-%m-%d %H:%M:%S"):
    """unix timestamp (ms) to isofromat datetime."""

    timestamp = int(timestamp) // 1000
    return datetime.datetime.fromtimestamp(timestamp).strftime(fmt)


class LazyProperty:

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None: return self
        value = self.func(instance)
        setattr(instance, self.func.__name__, value)
        return value


class Timer:

    def __init__(self, *, unit="s"):
        self.curr_elapsed = 0
        self.total_elapsed = 0
        self.start_time = 0
        self.count = 0
        self.is_running = False
        # unit只支持秒和毫秒
        self.unit = {"s": 1, "ms": 1000}[unit]

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.is_running = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        self.curr_elapsed = elapsed * self.unit
        self.total_elapsed += self.curr_elapsed
        self.is_running = False
        self.count += 1

    def reset(self):
        self.curr_elapsed = 0
        self.total_elapsed = 0
        self.start_time = 0
        self.count = 0
        self.is_running = False

    @property
    def average(self):
        if not self.count: return -1
        return self.total_elapsed / self.count


def timing(timer_name):
    """对类的方法进行timing. timer_name为类中timer的名字."""

    # yapf: disable
    def decorate(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            timer = getattr(self, timer_name)
            # 这里考虑到类继承中override的情况
            if timer.is_running:
                return func(self, *args, **kwargs)
            with timer:
                return func(self, *args, **kwargs)
        return wrapper

    return decorate
    # yapf: enable


class UnicodeString:
    """增加f-string对中文的支持."""

    _placeholders = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self, text):
        self.text = text
        self.width = sum(2 - int(c.isascii()) for c in text)

    def __str__(self):
        return self.text

    def __format__(self, spec=None):
        if spec is None: return self.text
        if self.width == len(self.text):
            return format(self.text, spec)
        repeat = self.width // len(self._placeholders)
        rest = self.width % len(self._placeholders)
        text = self._placeholders * repeat + self._placeholders[:rest]
        return format(text, spec).replace(text, self.text)


if __name__ == "__main__":
    pass
