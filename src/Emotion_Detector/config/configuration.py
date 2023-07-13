from Emotion_Detector.utils import *
from Emotion_Detector.constants import *
from Emotion_Detector.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig)


class ConfigurationManager:
    def __init__(
            self,
            config_path = CONFIG_FILE_PATH,
            params_path = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])
        create_directories([config.local_data_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_url = config.source_url,
            local_data_dir = config.local_data_dir,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )
        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing


        create_directories([config.root_dir])
        create_directories([config.train_tf_records_dir])
        create_directories([config.test_tf_records_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir = Path(config.root_dir),
            train_dir = Path(config.train_dir),
            val_dir = Path(config.val_dir),
            train_tf_records_dir = Path(config.train_tf_records_dir),
            test_tf_records_dir = Path(config.test_tf_records_dir),
            train_tf_records_file_path = config.train_tf_records_file_path,
            test_tf_records_file_path = config.test_tf_records_file_path,
            params_batch_size = self.params.BATCH_SIZE,
            params_class_names = self.params.CLASS_NAMES,
            params_image_size = self.params.IMAGE_SIZE,
            params_shuffle = self.params.SHUFFLE,
            params_random_seed = self.params.RANDOM_SEED,
            params_train_num_shards = self.params.TRAIN_NUM_SHARDS,
            params_test_num_shards = self.params.TEST_NUM_SHARDS,
            param_random_rotation_left_factor = self.params.RANDOM_ROTATION_LEFT_FACTOR,
            param_random_rotation_right_factor = self.params.RANDOM_ROTATION_RIGHT_FACTOR,
            param_random_flip_mode = self.params.RANDOM_FLIP_MODE,
            param_random_contrast_factor = self.params.RANDOM_CONTRAST_FACTOR
        )
        return data_preprocessing_config