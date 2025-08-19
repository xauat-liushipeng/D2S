"""
Logging utilities for D2S training
"""
import os
import sys
import logging
from datetime import datetime
from typing import Optional


class TeeLogger:
    """
    Logger that outputs to both console and file
    """
    def __init__(self, log_file: str, console_level: int = logging.INFO, file_level: int = logging.INFO):
        self.log_file = log_file
        self.console_level = console_level
        self.file_level = file_level
        
        # Create logger
        self.logger = logging.getLogger('D2S')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # Clear existing handlers
        
        # Create formatters
        console_formatter = logging.Formatter('%(message)s')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(file_level)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, msg: str):
        """Log info message"""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message"""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message"""
        self.logger.error(msg)
    
    def debug(self, msg: str):
        """Log debug message"""
        self.logger.debug(msg)
    
    def critical(self, msg: str):
        """Log critical message"""
        self.logger.critical(msg)
    
    def print(self, msg: str):
        """Print message (same as info)"""
        self.logger.info(msg)


def create_work_dir(base_dir: str = "work_dir") -> str:
    """
    Create work directory with date format: YYYYMMDD_N
    
    Args:
        base_dir: Base directory name
        
    Returns:
        Created work directory path
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    today = datetime.now().strftime("%Y%m%d")
    
    # Find next available number for today
    counter = 1
    while True:
        work_dir = os.path.join(base_dir, f"{today}_{counter}")
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
            break
        counter += 1
    
    return work_dir


def setup_logging(work_dir: str, log_name: str = "training.log") -> TeeLogger:
    """
    Setup logging for training
    
    Args:
        work_dir: Work directory path
        log_name: Log file name
        
    Returns:
        Configured logger
    """
    log_file = os.path.join(work_dir, log_name)
    return TeeLogger(log_file)
