from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from Emotion_Detector.utils import *
from Emotion_Detector.utils.model_store import LenetModel
from Emotion_Detector.entity.config_entity import ModelTrainingConfig

class Model_Training:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def _get_data(self):
        train_data = reconstruct_data_from_tfrecords(path = self.config.train_tf_records_file_path,
                                                    num_shards= self.config.param_train_num_shards, 
                                                    batch_size = self.config.param_batch_size )
        val_data = reconstruct_data_from_tfrecords(path = self.config.test_tf_records_file_path,
                                                    num_shards= self.config.param_test_num_shards, 
                                                    batch_size = self.config.param_batch_size )
        return train_data, val_data
        
    def _get_model(self):
        input_shape = (1, self.config.param_image_size, self.config.param_image_size, 3)
        print(self.config.all_params)
        lenet_model = LenetModel(configurations = self.config.all_params)
        lenet_model.build(input_shape = input_shape)
        return lenet_model
    
    def train_model(self):
        logger.info(f"Reconstructing the data from tensorflow records...")
        train_data, val_data = self._get_data()
        logger.info(f"Data loaded successfully.")

        logger.info(f"Geting the Sub Classed Lenet Model...")
        model = self._get_model()
        logger.info(f"Model Loaded Successfully...")

        logger.info(f"Preparing Checkpoint Callback...")
        checkpoint_callback = ModelCheckpoint(
            filepath= self.config.checkpoints_file_path,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )

        logger.info(f"Preparing Tensorboard Callback...")
        tensorboard_callback = TensorBoard(log_dir=self.config.tensorboard_log_dir, histogram_freq=1)


        loss_function = SparseCategoricalCrossentropy()
        metrics = [SparseCategoricalAccuracy(name = "accuracy")]

        logger.info(f"Compiling the model...")
        model.compile(
                optimizer = Adam(learning_rate= self.config.param_learning_rate),
                loss = loss_function,
                metrics = metrics
            )
        
        logger.info(f"Starting training with {self.config.param_epochs} epochs and validation data...")
        history = model.fit(train_data, epochs = self.config.param_epochs,validation_data = val_data, callbacks = [checkpoint_callback, tensorboard_callback], verbose = 1)
        logger.info(f"Model Training Completed Successfully.")
        logger.info(f"Model Summary /n {model.summary()}")

        logger.info(f"Saving the model at {self.config.model_dir}...")
        model.save(self.config.model_dir, save_format = 'tf')
        logger.info(f"Model successfully saved at {self.config.model_dir}.")

        logger.info(f"Saving the model history at {self.config.model_history_file_path}...")
        save_json(path = self.config.model_history_file_path, data = history.history)
        logger.info(f"Model history successfully saved at {self.config.model_history_file_path}.")