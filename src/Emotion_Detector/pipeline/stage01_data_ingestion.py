from Emotion_Detector.components import DataIngestion
from Emotion_Detector.config import ConfigurationManager
from Emotion_Detector import logger

def main():
    try:
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config = data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.unzip_and_clean()
    except Exception as e:
        raise e


if __name__ == '__main__':
    try:
        logger.info(f"{'>>>>'*5} Starting Data Ingestion. {'<<<<'*5}.")
        main()
        logger.info(f"{'>>>>'*5} Data Ingestion Stage Completed! {'<<<<'*5}.")
    except Exception as e:
        raise e
