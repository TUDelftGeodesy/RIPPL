import logging
import os

# First remove the debug log from earlier runs
if os.path.exists('debug.log'):
    os.remove('debug.log')

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)