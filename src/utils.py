import os
import logging
import json
import pandas as pd

def setup_logging(log_file="pipeline.log"):
    """Sets up professional logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized.")

def ensure_dirs(dirs=None):
    """Ensures that all specified directories exist."""
    if dirs is None:
        dirs = [
            "datasets",
            "notebooks",
            "src",
            "outputs/figures",
            "outputs/tables",
            "outputs/metrics",
            "reports"
        ]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            logging.info(f"Created directory: {d}")
        else:
            logging.info(f"Directory exists: {d}")

def save_json(data, filepath):
    """Saves a dictionary as a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, default=str)
    logging.info(f"Saved JSON data to {filepath}")

def load_json(filepath):
    """Loads JSON data from a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_table(df, filepath_csv, filepath_markdown=None):
    """Saves a pandas DataFrame to CSV and optionally markdown."""
    df.to_csv(filepath_csv, index=False)
    logging.info(f"Saved CSV table to {filepath_csv}")
    if filepath_markdown:
        with open(filepath_markdown, 'w', encoding='utf-8') as f:
            f.write(df.to_markdown(index=False))
        logging.info(f"Saved Markdown table to {filepath_markdown}")
