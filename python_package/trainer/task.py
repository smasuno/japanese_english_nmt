import hypertune
import argparse
import hypertune

args_parser = argparse.ArgumentParser()

# Estimator arguments
args_parser.add_argument(
    '--learning-rate',
    help='Learning rate value for the optimizers.',
    default=2e-5,
    type=float)
args_parser.add_argument(
    '--weight-decay',
    help="""
    The factor by which the learning rate should decay by the end of the
    training.

    decayed_learning_rate =
    learning_rate * decay_rate ^ (global_step / decay_steps)

    If set to 0 (default), then no decay will occur.
    If set to 0.5, then the learning rate should reach 0.5 of its original
        value at the end of the training.
    Note that decay_steps is set to train_steps.
    """,
    default=0.01,
    type=float)

# Enable hyperparameter
args_parser.add_argument(
    '--hp-tune',
    default="n",
    help='Enable hyperparameter tuning. Valida values are: "y" - enable, "n" - disable')

