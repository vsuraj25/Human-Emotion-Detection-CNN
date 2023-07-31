from Emotion_Detector.components import Resnet_Model_Training
from Emotion_Detector.config import ConfigurationManager
from Emotion_Detector import logger

def main():
    try:
        config = ConfigurationManager()
        resnet_model_training_config = config.get_resnet_model_training_config()
        model_trainer = Resnet_Model_Training(config = resnet_model_training_config)
        model_trainer.train_model() 
    except Exception as e:
        raise e


if __name__ == '__main__':
    try:
        logger.info(f"{'>>>>'*5} Starting Resnet Model Training. {'<<<<'*5}.")
        main()
        logger.info(f"{'>>>>'*5} Resnet Model Training Stage Completed! {'<<<<'*5}.")
    except Exception as e:
        raise e
