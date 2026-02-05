import logging
import sys
import datetime
import torch
import os
import random
import numpy as np
from body_combined import train_combined, predict_combined


def setup_logger(log_file_path):
    """
    Set up logger to write all output to both console and log file
    """
    # Create logger
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def main():
    # Set CuBLAS workspace config for deterministic behavior (must be set before any CUDA operations)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['TRANSFORMERS_CACHE'] = ''
    prop = "Tc_supercon"
    
    # Set seed FIRST, before any random operations
    SEED = 42 
    set_seed(SEED)
    
    # Set up logging with timestamp in filename
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file_path = f"./logs/training_log_{timestamp}.log"
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Setup logger
    logger = setup_logger(log_file_path)
    
    logger.info("=" * 50)
    logger.info("TRAINING SESSION STARTED")
    logger.info("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device is: {device}")
    logger.info("Training started")
    
    try:
        model_file = train_combined(prop=prop, logger=logger, timestamp=timestamp)
        logger.info("Training completed successfully")
        predict_combined(prop=prop, model_path=model_file, logger=logger)
        logger.info("Prediction completed successfully")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.exception("Full traceback:")
        raise
    
    logger.info("=" * 50)
    logger.info("SESSION COMPLETED")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
