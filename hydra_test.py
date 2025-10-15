import os

import hydra
from omegaconf import DictConfig, OmegaConf

# python my_app.py --multirun '+experiment=glob(*)' can run multiple configs


@hydra.main(config_path="conf", config_name="testconf", version_base=None)
def main(cfg: DictConfig) -> None:
    print(f"Working directory : {os.getcwd()}")
    print(
        f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
