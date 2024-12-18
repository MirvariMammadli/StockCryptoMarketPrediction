from classes.Config import Config
from classes.FinancialDataPipeline import FinancialDataPipeline
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning)  # Suppress user warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress future warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

if __name__ == "__main__":
    config_data = {
        "reddit_client_id": "MmsK7PahE1pwyzbLF2wBKQ",
        "reddit_client_secret": "zAFr9cMvdk1TacfbKiuLL2HH5I0X6A",
        "reddit_user_agent": "mmmirvari",
    }
    config = Config(config_data)
    pipeline = FinancialDataPipeline(config)
    pipeline.run()
