from Emotion_Detector.components import Model_Training
from Emotion_Detector.config import ConfigurationManager
from Emotion_Detector import logger

def main():
    try:
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_trainer = Model_Training(config = model_training_config)
        model_trainer.train_model()
    except Exception as e:
        raise e


if __name__ == '__main__':
    try:
        logger.info(f"{'>>>>'*5} Starting Model Training. {'<<<<'*5}.")
        main()
        logger.info(f"{'>>>>'*5} Model Training Stage Completed! {'<<<<'*5}.")
    except Exception as e:
        raise e
