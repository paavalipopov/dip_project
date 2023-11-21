import warnings
warnings.filterwarnings("ignore")

import math, re, os
import tensorflow as tf
import numpy as np
import glob

from tqdm import tqdm

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [SIZE, SIZE, 3]) # explicit size needed for TPU
    return image


def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum # returns a dataset of image(s)


def do_stuff():
    print(f"Converting size {SIZE}")
    train_paths = glob.glob(f"./assets/data/original_data/tfrecords-jpeg-{SIZE}x{SIZE}/train/*")
    val_paths = glob.glob(f"./assets/data/original_data/tfrecords-jpeg-{SIZE}x{SIZE}/val/*")
    test_paths = glob.glob(f"./assets/data/original_data/tfrecords-jpeg-{SIZE}x{SIZE}/test/*")

    print("Processing train data")
    train_dataset = tf.data.TFRecordDataset(train_paths).map(read_labeled_tfrecord)
    train_save_path = f"./assets/data/np_data/{SIZE}x{SIZE}/train"
    os.makedirs(train_save_path, exist_ok=True)
    train_data = {class_idx: [] for class_idx in range(104)}

    for example in tqdm(train_dataset):
        train_data[example[1].numpy()].append(example[0].numpy())

    for class_idx in tqdm(range(104)):
        class_data = np.stack(train_data[class_idx])
        labels = np.array([class_idx]*class_data.shape[0])
        
        np.savez(f"{train_save_path}/class_{class_idx}.npz", data=class_data, labels=labels)

    print("Processing val data")
    val_dataset = tf.data.TFRecordDataset(val_paths).map(read_labeled_tfrecord)
    val_save_path = f"./assets/data/np_data/{SIZE}x{SIZE}/val"
    os.makedirs(val_save_path, exist_ok=True)
    val_data = {class_idx: [] for class_idx in range(104)}

    for example in tqdm(val_dataset):
        val_data[example[1].numpy()].append(example[0].numpy())

    for class_idx in tqdm(range(104)):
        class_data = np.stack(val_data[class_idx])
        labels = np.array([class_idx]*class_data.shape[0])

        np.savez(f"{val_save_path}/class_{class_idx}.npz", data=class_data, labels=labels)

    print("Processing test data")
    test_dataset = tf.data.TFRecordDataset(test_paths).map(read_unlabeled_tfrecord)
    test_save_path = f"./assets/data/np_data/{SIZE}x{SIZE}/test"
    os.makedirs(test_save_path, exist_ok=True)
    test_data = []
    test_idx = []
    for example in tqdm(test_dataset):
        test_data.append(example[0].numpy())
        test_idx.append(example[1].numpy())
    
    test_data = np.stack(test_data)
    test_idx = np.stack(test_idx)
    np.savez(f"{test_save_path}/data.npz", data=test_data, id=test_idx)

if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--res", 
        "-r", 
        help="Resolution",
        type=int,
        choices=[192, 224, 331, 512],
        default=512,
    )
    SIZE = parser.parse_args().res

    do_stuff()