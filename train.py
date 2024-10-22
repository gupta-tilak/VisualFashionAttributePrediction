from python_code.iMatDataset import iMatDataModule
from python_code.AttrPredModel import AttrPred_Resnet50
import pytorch_lightning as pl
from argparse import ArgumentParser
from torchvision.transforms import *

if __name__ == "__main__":
    # Initialize argument parser
    parser = ArgumentParser()

    # Add common arguments (e.g., description)
    parser.add_argument("--description", type=str, required=False, help="Description")
    
    # Manually add Trainer-related arguments
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use (0 for CPU)")
    parser.add_argument("--precision", type=int, default=32, help="Precision (16/32-bit)")

    # Add model-specific arguments
    parser = AttrPred_Resnet50.add_model_specific_args(parser)

    # Add dataset-specific arguments
    parser = iMatDataModule.add_dataset_specific_args(parser)

    # Parse all arguments
    args = parser.parse_args()

    # Define image augmentations
    image_augmentations = [ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), RandomHorizontalFlip()]

    # Initialize the data module with parsed arguments
    dm = iMatDataModule(**vars(args))
    dm.prepare_data()
    dm.setup()

    # Initialize the model with parsed arguments
    model = AttrPred_Resnet50(102, **vars(args))

    # Initialize the Trainer with the parsed arguments
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        # gpus=args.gpus,
        precision=args.precision,
        callbacks=[model.checkpoint]  # Pass the checkpoint callback from the model
    )

    # Start training
    trainer.fit(model, datamodule=dm)  # Pass the data module directly here
