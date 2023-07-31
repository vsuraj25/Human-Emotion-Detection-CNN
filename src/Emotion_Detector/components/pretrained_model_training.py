from Emotion_Detector.utils import *
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from Emotion_Detector.config.configuration import PretrainedModelTrainingConfig
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, BatchNormalization
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class Pretrained_Model_Training:
    def __init__(self, config: PretrainedModelTrainingConfig):
        self.config = config

    def _get_data(self):
        train_data, val_data = get_augmented_data(train_dir_path= self.config.train_dir,
                                                  val_dir_path= self.config.val_dir,
                                                  params= self.config.all_params
                                                  )
        return train_data, val_data
        
    def _get_model(self):
        backbone = tf.keras.applications.efficientnet.EfficientNetB4(
            include_top= self.config.param_include_top,
            weights='imagenet',
            input_shape=(self.config.all_params['IMAGE_SIZE'], self.config.all_params['IMAGE_SIZE'], 3)
        )
        backbone.trainable = False
        model = tf.keras.Sequential([
            Input(shape = (self.config.all_params['IMAGE_SIZE'], self.config.all_params['IMAGE_SIZE'], 3)),
            backbone,
            GlobalAveragePooling2D(),
            Dense(self.config.param_en_dense_1, activation = "relu"),
            BatchNormalization(),
            Dense(self.config.param_en_dense_2, activation = "relu"),
            Dense(self.config.all_params["NUM_CLASSES"], activation = "softmax"),
        ])

        return model
    
    def train_model(self):
        
        logger.info(f"Reconstructing the data from tensorflow records...")
        train_data, val_data = self._get_data()
        logger.info(f"Data loaded successfully.")

        logger.info(f"Geting the Pretrained Model...")
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


        loss_function = CategoricalCrossentropy()
        metrics = [CategoricalAccuracy(name = "accuracy")]

        logger.info(f"Compiling the model...")
        model.compile(
                optimizer = Adam(learning_rate= 0.001),
                loss = loss_function,
                metrics = metrics
            )
        
        logger.info(f"Starting training with {self.config.param_en_epoch} epochs and validation data...")
        history = model.fit(train_data,batch_size = self.config.param_batch_size, epochs = self.config.param_en_epoch,validation_data = val_data, callbacks = [checkpoint_callback, tensorboard_callback], verbose = 1)
        logger.info(f"Model Training Completed Successfully.")
        logger.info(f"Model Summary /n {model.summary()}")

        logger.info(f"Saving the model at {self.config.model_dir}...")
        model.save(self.config.model_dir, save_format = 'tf')
        logger.info(f"Model successfully saved at {self.config.model_dir}.")

        logger.info(f"Saving the model history at {self.config.model_history_file_path}...")
        save_json(path = self.config.model_history_file_path, data = history.history)
        logger.info(f"Model history successfully saved at {self.config.model_history_file_path}.")

        logger.info(f"Saving the model accuracy plot at {self.config.model_accuracy_plot_path}...")
        save_plt_fig(x = history.history['accuracy'],
                     y = history.history['val_accuracy'],
                     title = "Model Accuracy",
                     xlabel ='Epochs',
                     ylabel = "Accuracy",
                     legends= ['Train', 'Validation'],
                     fig_path = self.config.model_accuracy_plot_path)
        
        logger.info(f"Saving the model loss plot at {self.config.model_loss_plot_path}...")
        save_plt_fig(x = history.history['loss'],
                     y = history.history['val_loss'],
                     title = "Model loss",
                     xlabel ='Epochs',
                     ylabel = "Loss",
                     legends= ['Train', 'Validation'],
                     fig_path = self.config.model_loss_plot_path)
        logger.info(f"Model results saved successfully.")