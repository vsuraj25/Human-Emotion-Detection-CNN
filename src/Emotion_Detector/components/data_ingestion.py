import os
from pathlib import Path
from Emotion_Detector.entity.config_entity import DataIngestionConfig
from Emotion_Detector.utils import *
import urllib.request as request
from zipfile import ZipFile
from Emotion_Detector import logger
from tqdm import tqdm



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        logger.info(f"Getting Configurations...")
        self.config = config

    def download_file(self):
        logger.info(f"Downloading the zip file from the url - {self.config.source_url} \
                    to file - {self.config.local_data_file} ...")
        if not os.path.exists(self.config.local_data_file):
            filename, headers= request.urlretrieve(
                url = self.config.source_url,
                filename = self.config.local_data_file
            )
            logger.info(f"Zip File Downloaded and saved at {self.config.local_data_file}.")
        else:
            logger.info(f" File already present in {self.config.local_data_file} of size\
                         {get_size(Path(self.config.local_data_file))}.")

    def _preprocess(self, zf : ZipFile, f : str, working_dir : str):
        target_filepath = os.path.join(working_dir, f)
        if not os.path.exists(target_filepath):
            zf.extract(f, working_dir)
        
        if os.path.getsize(target_filepath) == 0:
            os.remove(target_filepath)

    def unzip_and_clean(self):
        logger.info(f"Extracting the zip file...")
        with ZipFile(file = self.config.local_data_file, mode="r") as zf:
            list_of_files = zf.namelist()
            for f in tqdm(list_of_files):
                self._preprocess(zf, f, self.config.unzip_dir)
        logger.info(f"Successfully extracted data from the zip file and saved a location - \
                    {self.config.unzip_dir}")