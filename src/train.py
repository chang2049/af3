import argparse
import torch
from torch.utils.data import DataLoader
from alphafold3_pytorch import Alphafold3, alphafold3_inputs_to_batched_atom_input
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


def train(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/alphafold3_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # Initialize dataset and data loader
    train_dataset = PDBDataset(args.data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Initialize model
    model = Alphafold3(
        dim_atom_inputs=args.dim_atom_inputs,
        dim_template_feats=args.dim_template_feats,
        # Add other model parameters as needed
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # Convert data to the format required by the model
            inputs = alphafold3_inputs_to_batched_atom_input(batch, atoms_per_window=args.atoms_per_window)

            # Move inputs to the specified device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in
                      inputs.model_forward_dict().items()}

            # Forward pass
            loss = model(**inputs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Log batch loss
            writer.add_scalar('Loss/batch', loss.item(), epoch * len(train_loader) + batch_idx)

            if (batch_idx + 1) % args.log_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Log epoch loss
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Average Loss: {avg_loss:.4f}")

        # Update learning rate
        scheduler.step(avg_loss)

        # Save the model if it's the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, args.model_dir, 'best_model.pth')

        # Save checkpoint every few epochs
        if (epoch + 1) % args.save_interval == 0:
            save_model(model, args.model_dir, f'checkpoint_epoch_{epoch + 1}.pth')

    # Save the final model
    save_model(model, args.model_dir, 'final_model.pth')

    writer.close()
    print("Training completed.")


def save_model(model, model_dir, filename):
    """
    Save the model to a file.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filepath = os.path.join(model_dir, filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.config  # Assuming the model has a config attribute
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(model_path, device):
    """
    Load a saved model.
    """
    checkpoint = torch.load(model_path, map_location=device)
    model = Alphafold3(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Alphafold3 model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dim_atom_inputs", type=int, default=77, help="Dimension of atom inputs")
    parser.add_argument("--dim_template_feats", type=int, default=108, help="Dimension of template features")
    parser.add_argument("--atoms_per_window", type=int, default=27, help="Number of atoms per window")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--log_interval", type=int, default=10, help="How often to log training status")
    parser.add_argument("--save_interval", type=int, default=5, help="How often to save model checkpoints")

    args = parser.parse_args()
    train(args)