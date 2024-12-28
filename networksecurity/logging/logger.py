import logging
import os
from datetime import datetime

# Generate log file name
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"

# Define the directory where logs will be stored
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Ensure the directory exists
os.makedirs(logs_path, exist_ok=True)

# Full log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("NetworkSecurityLogger")