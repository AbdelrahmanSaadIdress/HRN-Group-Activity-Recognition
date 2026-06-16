import argparse
from scripts.B1_NoRelations import (
    train_stage_one,
    test_stage_one,
    train_stage_two,
    test_stage_two,
)
from utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="HRN Group Activity Recognition")

    parser.add_argument("--mode",       type=str, required=True,  choices=["train", "test"],
                        help="train or test")
    parser.add_argument("--model",      type=str, required=True,  choices=["B1_NoRelations"],
                        help="which model to run")
    parser.add_argument("--stage",      type=int, required=True,  choices=[1, 2],
                        help="which stage of the model")
    parser.add_argument("--config",     type=str, required=True,
                        help="path to the YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint path: resume path for training, model path for testing")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="path to stage 1 best checkpoint (required for stage 2 training)")

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.model == "B1_NoRelations":

        if args.mode == "train" and args.stage == 1:
            train_stage_one(config, checkpoint_path=args.checkpoint)

        elif args.mode == "test" and args.stage == 1:
            if args.checkpoint is None:
                raise ValueError("--checkpoint is required for testing.")
            test_stage_one(config, checkpoint_path=args.checkpoint)

        elif args.mode == "train" and args.stage == 2:
            if args.stage1_checkpoint is None:
                raise ValueError("--stage1_checkpoint is required for stage 2 training.")
            train_stage_two(
                config,
                stage1_checkpoint=args.stage1_checkpoint,
                checkpoint_path=args.checkpoint,
            )

        elif args.mode == "test" and args.stage == 2:
            if args.checkpoint is None:
                raise ValueError("--checkpoint is required for testing.")
            test_stage_two(config, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()



