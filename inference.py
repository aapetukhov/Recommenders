import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.pipelines import run_offline_inference
from src.utils.io_utils import ROOT_PATH


def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )
    logging.getLogger("annoy").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(cfg: DictConfig) -> None:
    """
    Entry point for running offline inference and metric computation.
    """

    _configure_logging()

    logging.info("Starting offline inference with config:\n%s", OmegaConf.to_yaml(cfg))

    artifacts = run_offline_inference(cfg)

    logging.info("Finished inference. Aggregated metrics:")
    for metric_name, value in artifacts.metrics.items():
        logging.info("  %s = %.4f", metric_name, value)

    logging.info("Artifacts are stored under %s", _resolve_output_root(cfg))


def _resolve_output_root(cfg: DictConfig) -> Path:
    output_dir = cfg.inference.io.get("output_dir")
    if output_dir is None:
        return ROOT_PATH
    path = Path(output_dir)
    if not path.is_absolute():
        path = (ROOT_PATH / path).resolve()
    return path


if __name__ == "__main__":
    main()
