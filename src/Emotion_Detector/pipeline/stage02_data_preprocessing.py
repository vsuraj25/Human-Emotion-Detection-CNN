from Emotion_Detector.components import Data_Preprocessing
from Emotion_Detector.config import ConfigurationManager
from Emotion_Detector import logger

def main():
    try:
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = Data_Preprocessing(config = data_preprocessing_config)
        data_preprocessing.preprocess_and_create_tf_records()
    except Exception as e:
        raise e


if __name__ == '__main__':
    try:
        logger.info(f"{'>>>>'*5} Starting Data Preprocessing. {'<<<<'*5}.")
        main()
        logger.info(f"{'>>>>'*5} Data Preprocessing Stage Completed! {'<<<<'*5}.")
    except Exception as e:
        raise e
