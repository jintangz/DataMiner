import logging
import sys

logger = logging.getLogger()

logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler("../data_process.log", mode='w', encoding='utf8')
formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(file_handler)