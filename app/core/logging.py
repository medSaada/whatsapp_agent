import logging
import sys

# Set up a basic logger for the application
logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Utility function to get the logger
get_logger = lambda: logger 