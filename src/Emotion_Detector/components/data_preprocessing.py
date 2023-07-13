from Emotion_Detector.entity.config_entity import DataPreprocessingConfig
from Emotion_Detector.utils import *
import tensorflow as tf
from tensorflow.keras.layers import RandomFlip, RandomRotation,RandomContrast

class Data_Preprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    
    def _load_data_from_directory(self):
        train_data = tf.keras.utils.image_dataset_from_directory(
            self.config.train_dir,
            labels='inferred', ## consider file as class
            label_mode='categorical', ## factorized classes
            class_names=self.config.params_class_names, ## Defined class name as per subdirectories
            color_mode='rgb',
            batch_size=self.config.params_batch_size,
            image_size=(self.config.params_image_size, self.config.params_image_size),
            shuffle=self.config.params_shuffle,
            seed=self.config.params_random_seed
        )


        val_data = tf.keras.utils.image_dataset_from_directory(
            self.config.val_dir,
            labels='inferred', ## consider file as class
            label_mode='categorical', ## factorized classes
            class_names=self.config.params_class_names, ## Defined class name as per subdirectories
            color_mode='rgb',
            batch_size=self.config.params_batch_size,
            image_size=(self.config.params_image_size, self.config.params_image_size),
            shuffle=self.config.params_shuffle,
            seed=self.config.params_random_seed
        )

        return train_data, val_data
    
    
    def _augment_data(self, train_data):
        try:
            augment_layers = tf.keras.Sequential([
                RandomRotation(factor = (self.config.param_random_rotation_left_factor, self.config.param_random_rotation_right_factor)),
                RandomFlip(mode = self.config.param_random_flip_mode),
                RandomContrast(factor = self.config.param_random_contrast_factor)
            ])

            def augment_layer(image, label):
                return augment_layers(image, training = True), label
            
            train_data = (
                train_data.map(augment_layer, num_parallel_calls=tf.data.AUTOTUNE)#.prefetch(tf.data.AUTOTUNE)
            )
        except Exception as e:
            logger.info(e)

        return train_data
    
    def preprocess_and_create_tf_records(self):
        try:
            logger.info(f"Loading the ingested training and validation data...")
            train_data, val_data = self._load_data_from_directory()
            logger.info(f"Data Loaded Successfully!.")

            logger.info(f"Augmenting the train data...")
            train_data = self._augment_data(train_data)
            logger.info(f"Train data augmented!")

            logger.info(f"Unbatching the training and validation data... ")
            train_data = (train_data.unbatch())
            val_data = (val_data.unbatch())

            logger.info(f"Encoding the data from image to byte...")
            encoded_train_data = (train_data.map(image_to_byte_encoder))
            encoded_val_data = (val_data.map(image_to_byte_encoder))

            if (os.path.exists(self.config.train_tf_records_dir) and 
                len(os.listdir(self.config.train_tf_records_dir)) == self.config.params_train_num_shards):

                logger.info('Train tf records already exists!')

            else:
                logger.info('Creating tf records for train set...') 
                write_tf_records(NUM_SHARDS= self.config.params_train_num_shards,
                                encoded_data=encoded_train_data,
                                path_to_write=self.config.train_tf_records_file_path
                                )
                logger.info(f"tf records for training set created and stored at {self.config.train_tf_records_dir}...")
                
            if (os.path.exists(self.config.test_tf_records_dir) and 
                len(os.listdir(self.config.test_tf_records_dir)) == self.config.params_test_num_shards):

                logger.info('Test tf records already exists!') 
                
            else:
                logger.info('Creating tf records for test set...') 
                write_tf_records(NUM_SHARDS= self.config.params_test_num_shards,
                                encoded_data=encoded_val_data,
                                path_to_write=self.config.test_tf_records_file_path
                                )
                logger.info(f"tf records for validation set created and stored at {self.config.test_tf_records_dir}...")

            return "TF Records Created Successfully for training and validation data"
        except Exception as e:
            logger.info(e)
        