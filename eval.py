import os
import time

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import torch.multiprocessing

from datasets.dataloader import get_all_dl
from utils.custom_metrics import plot_distributions, save_scores, load_scores, get_scores
from utils.tools import (
    print_args,
    set_device,
    initialize_network,
    initialize_approach,
    initialize_logger,
    seed_everything
)

torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(config_path="./conf", config_name="config")
def run(cfg: DictConfig) -> None:

    # --- Misc
    torch.cuda.empty_cache()
    t_start = time.time()

    # --- Set paths
    cfg.original_cwd = get_original_cwd()

    # --- Set seed
    seed_everything(seed=cfg.seed)

    # --- Args -- CUDA
    set_device(cfg)

    # --- Initialize logger
    full_run_name, logger = initialize_logger(cfg)

    # --- Print initial arguments
    print_args(cfg, logger)

    # --- Loaders
    in_train_dl, in_val_dl, in_test_dl, out_seen_train_dl, out_seen_val_dl, out_seen_test_dl, out_unseen_test_dl_dict = get_all_dl(cfg,
                                                                                                                    logger)

    # --- Network
    net = initialize_network(cfg, in_test_dl)

    # --- OoD Approach
    appr = initialize_approach(cfg, net, logger)

    # --- Setup
    appr.setup()

    # --- Get scores
    in_test_scores, out_seen_test_scores, out_unseen_test_scores_dict, _ = appr.get_test_scores(in_test_dl, out_seen_test_dl, out_unseen_test_dl_dict)

    # --- Get metrics
    full_metrics_dict, full_aux_metrics_dict = appr.get_all_metrics(in_test_scores, out_seen_test_scores,
                                                                    out_unseen_test_scores_dict)

    plot_distributions(full_run_name, full_aux_metrics_dict, cfg)

    # --- Log results
    logger.log_results(full_metrics_dict, appr, full_run_name, t_start)

if __name__ == '__main__':
    run()
