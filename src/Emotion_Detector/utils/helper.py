import os
from box.exceptions import BoxValueError
import yaml
from Emotion_Detector import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import tensorflow as tf
from tensorflow.train import BytesList, FloatList, Int64List, Example, Features, Feature


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns
    Args:
        path_to_yaml (str): path like input
    Raises:
        ValueError: if yaml file is empty
        e: empty file
    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories
    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data
    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data
    Args:
        path (Path): path to json file
    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file
    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data
    Args:
        path (Path): path to binary file
    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB
    Args:
        path (Path): path of the file
    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

## Encoding the image as byte
def image_to_byte_encoder(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = tf.io.encode_jpeg(image)
    return image, tf.argmax(label)



def write_tf_records(NUM_SHARDS, encoded_data, path_to_write):

    ## Serializing the image as bytes feature
    def create_example(image, label):

        bytes_feature = Feature(
            bytes_list=BytesList(value=[image]))

        int_feature = Feature(
            int64_list=Int64List(value=[label]))

        example = tf.train.Example(
            features=Features(feature={
                'images': bytes_feature,
                'labels': int_feature,
            }))
        
        return example.SerializeToString()
    
    for shard_number in range(NUM_SHARDS):
        sharded_data = (
            encoded_data.shard(NUM_SHARDS, shard_number).as_numpy_iterator()
        )

        with tf.io.TFRecordWriter(path_to_write.format(shard_number)) as file_writer:
            for encoded_image, encoded_label in sharded_data:
                example = create_example(encoded_image, encoded_label)
                file_writer.write(example)


def reconstruct_data_from_tfrecords(path, num_shards, batch_size):

    def parse_tfrecords(example):

        feature_description = {
            "images" : tf.io.FixedLenFeature([], tf.string),
            "labels" : tf.io.FixedLenFeature([], tf.int64)
        }
        example = tf.io.parse_single_example(example, feature_description)
        example['images'] = tf.image.convert_image_dtype(
            tf.io.decode_jpeg(example["images"], channels = 3), dtype = tf.float32)

        return example["images"], example["labels"]


    recons_data = tf.data.TFRecordDataset(filenames= [path.format(p)  for p in range(num_shards)])

    parsed_data = (recons_data.map(parse_tfrecords).batch(batch_size).prefetch(tf.data.AUTOTUNE))

    return parsed_data

    
