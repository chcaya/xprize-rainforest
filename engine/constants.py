from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

root_dir = os.getenv('LOCAL_PATH')
data_paths = {
    'quebec': Path(root_dir + 'quebec_trees')
}
