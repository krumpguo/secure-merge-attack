import os
import argparse

import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. "
    )
    parser.add_argument(
        "--dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default='/gscratch/efml/gamaga/.cache/open_clip',
        help='Directory for caching models from OpenCLIP'
    )
    parser.add_argument(
        "--device_num",
        type=int,
        default=0
    )

    parser.add_argument("--lambda_adv", type=float, default=1.0, help="outer: weight for anti-merge term")
    parser.add_argument("--attacker_lr", type=float, default=1e-2, help="inner attacker lr")
    parser.add_argument("--attacker_steps", type=int, default=1, help="inner steps per batch")
    parser.add_argument("--protected_last_k_blocks", type=int, default=4, help="only protect last K transformer blocks")

    # policy A: block gating
    parser.add_argument("--use_policy_A", type=int, default=1, help="use block-gating policy A")
    parser.add_argument("--temp_A", type=float, default=1.0, help="sigmoid temperature for policy A gates")

    # policy B: low-rank projection selector
    parser.add_argument("--use_policy_B", type=int, default=1, help="use low-rank policy B")
    parser.add_argument("--rank_r", type=int, default=4, help="rank for policy B")
    parser.add_argument("--temp_B", type=float, default=1.0, help="sigmoid temperature for policy B gates")

    # policy C: basis mixing (synthetic basis for generality)
    parser.add_argument("--use_policy_C", type=int, default=1, help="use basis-mixing policy C")
    parser.add_argument("--basis_k", type=int, default=4, help="number of synthetic basis directions per param for policy C")

    # softmin for selecting best attacker among A/B/C
    parser.add_argument("--softmin_tau", type=float, default=0.1, help="tau for softmin over policy losses (smaller->closer to min)")

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
