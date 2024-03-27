from config.config_parsers.xprize_parsers import XPrizeConfig
from engine.xprize.infer_pipeline import XPrizePipeline


def xprize_main(config: XPrizeConfig):
    pipeline = XPrizePipeline.from_config(config)
    pipeline.run()
