from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_dir: Path
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    train_dir: Path
    val_dir: Path
    train_tf_records_dir : Path
    test_tf_records_dir : Path
    train_tf_records_file_path : str
    test_tf_records_file_path : str
    params_batch_size : int
    params_class_names : list
    params_image_size : int
    params_shuffle : bool
    params_random_seed : int
    params_train_num_shards : int
    params_test_num_shards : int
    param_random_rotation_left_factor: float
    param_random_rotation_right_factor: float
    param_random_flip_mode: str
    param_random_translation_height_left_factor: float
    param_random_translation_height_right_factor: float
    param_random_translation_width_left_factor: float
    param_random_translation_width_right_factor: float
    param_random_zoom_factor: float

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    checkpoints_callback_dir : Path
    tensorboard_log_dir : Path
    model_history_dir : Path
    model_dir : Path
    model_file_path : str
    checkpoints_file_path : str
    model_history_file_path : Path
    model_accuracy_plot_path : Path
    model_loss_plot_path : Path
    train_dir: Path
    val_dir: Path
    param_image_size : int
    param_learning_rate : float
    param_epochs : int
    param_batch_size : int
    all_params : dict

@dataclass(frozen=True)
class ResnetModelTrainingConfig:
    root_dir: Path
    checkpoints_callback_dir : Path
    tensorboard_log_dir : Path
    model_history_dir : Path
    model_dir : Path
    model_file_path : str
    checkpoints_file_path : str
    model_history_file_path : Path
    model_accuracy_plot_path : Path
    model_loss_plot_path : Path
    train_dir: Path
    val_dir: Path
    param_resnet_epoch: int
    param_batch_size: int
    all_params : dict

@dataclass(frozen=True)
class PretrainedModelTrainingConfig:
    root_dir: Path
    checkpoints_callback_dir : Path
    tensorboard_log_dir : Path
    model_history_dir : Path
    model_dir : Path
    model_file_path : str
    checkpoints_file_path : str
    model_history_file_path : Path
    model_accuracy_plot_path : Path
    model_loss_plot_path : Path
    train_dir: Path
    val_dir: Path
    param_en_epoch: int
    param_batch_size: int
    param_include_top: bool
    param_en_dense_1: int
    param_en_dense_2: int
    all_params : dict