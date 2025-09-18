import os
import sys

import logging
from datetime import datetime


class TeeLogger:
    def __init__(self,
                 log_file: str,
                 console_level: int = logging.DEBUG,
                 file_level: int = logging.DEBUG
                 ):
        self.log_file = log_file
        self.console_level = console_level
        self.file_level = file_level

        # Create logger
        self.logger = logging.getLogger('D2S')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.propagate = False

        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(file_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(file_level)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def critical(self, msg: str):
        self.logger.critical(msg)


def create_work_dir(base_dir: str = "work_dir") -> str:
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    today = datetime.now().strftime("%Y%m%d")
    
    counter = 1
    while True:
        work_dir = os.path.join(base_dir, f"{today}_{counter}")
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
            break
        counter += 1
    
    return work_dir


def setup_logging(work_dir: str, log_name: str = "training.log") -> TeeLogger:
    log_file = os.path.join(work_dir, log_name)
    return TeeLogger(log_file)
