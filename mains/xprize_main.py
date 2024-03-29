from pathlib import Path

from config.config_parsers.xprize_parsers import XPrizeIOConfig
from engine.xprize.infer_pipeline import XPrizePipeline


def xprize_main(config: XPrizeIOConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=False, parents=True)

    pipeline = XPrizePipeline.from_config(config)
    pipeline.run()

    config.save_yaml_config(output_folder / 'xprize_config.yaml')
