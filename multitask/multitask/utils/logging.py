import logging


def create_logger(log_name, log_file):
    handler = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter(
        "[%(asctime)s, %(levelname)s] %(message)s", datefmt="%Y-%m-%d_%H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger
