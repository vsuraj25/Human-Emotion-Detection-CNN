from Emotion_Detector.components import Pretrained_Model_Training
from Emotion_Detector.config import ConfigurationManager
from Emotion_Detector import logger

def main():
    try:
        config = ConfigurationManager()
        pretrained_model_training_config = config.get_pretrained_model_training_config()
        model_trainer = Pretrained_Model_Training(config = pretrained_model_training_config)
        model_trainer.train_model() 
    except Exception as e:
        raise e


if __name__ == '__main__':
    try:
        logger.info(f"{'>>>>'*5} Starting Pretrained Model Training. {'<<<<'*5}.")
        main()
        logger.info(f"{'>>>>'*5} Pretrained Model Training Stage Completed! {'<<<<'*5}.")
    except Exception as e:
        raise e
