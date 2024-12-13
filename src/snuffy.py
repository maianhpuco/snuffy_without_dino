import argparse
import ast
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from lightly.utils.scheduler import CosineWarmupScheduler
from torch.autograd import Variable
from typing import Optional
from custom_layers import (
    MultiHeadedAttention, PositionwiseFeedForward, 
    BClassifier, FCLayer, MILNet, 
    VITFeatureExtractor, Encoder, EncoderLayer
)

device = "cuda" if torch.cuda.is_available() else "cpu"


class Snuffy(nn.Module):
    def __init__(self, args):
        super(Snuffy, self).__init__()
        self.args = args
        self.  # Instantiate the ViT feature extractor
        self.milnet = self._get_milnet()  # Get the MILNet for instance and bag classification
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

    def _get_milnet(self) -> nn.Module:
        """
        Creates the MILNet by combining the instance classifier and bag classifier.
        """
        vit_extractor = VITFeatureExtractor()
        i_classifier = FCLayer(in_size=self.args.feats_size, out_size=self.args.num_classes).to(device)
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.args.num_heads, self.args.feats_size).to(device)
        ff = PositionwiseFeedForward(self.args.feats_size, self.args.feats_size * self.args.mlp_multiplier,
                                     self.args.activation, self.args.encoder_dropout).to(device)

        b_classifier = BClassifier(
            Encoder(
                EncoderLayer(
                    self.args.feats_size,
                    c(attn),
                    c(ff),
                    self.args.encoder_dropout,
                    self.args.big_lambda,
                    self.args.random_patch_share
                ),
                self.args.depth
            ),
            self.args.num_classes,
            self.args.feats_size
        ).to(device)

        milnet = MILNet(
            vit_extractor, 
            i_classifier, 
            b_classifier
            ).to(device)

        init_funcs_registry = {
            'trunc_normal': nn.init.trunc_normal_,
            'kaiming_uniform': nn.init.kaiming_uniform_,
            'kaiming_normal': nn.init.kaiming_normal_,
            'xavier_uniform': nn.init.xavier_uniform_,
            'xavier_normal': nn.init.xavier_normal_,
            'orthogonal': nn.init.orthogonal_
        }

        modules = [(self.args.weight_init__weight_init_i__weight_init_b[1], 'i_classifier'),
                   (self.args.weight_init__weight_init_i__weight_init_b[2], 'b_classifier')]

        for init_func_name, module_name in modules:
            init_func = init_funcs_registry.get(init_func_name)
            for name, p in milnet.named_parameters():
                if p.dim() > 1 and name.split(".")[0] == module_name:
                    init_func(p)

        return milnet

    def forward(self, x):
        """
        Forward pass through the network.
        """
        ins_feats, bag_feats = self.vit_extractor(x)  # Feature extraction using ViT
        ins_prediction, bag_prediction, attentions = self.milnet(bag_feats)

        return ins_prediction, bag_prediction, attentions

    def compute_loss(self, bag_prediction: torch.Tensor, bag_label: torch.Tensor, ins_prediction: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the bag prediction and the label.
        """
        # Bag loss
        bag_loss = self.criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))

        # Instance loss (Max prediction)
        if len(ins_prediction.shape) == 2:
            max_prediction, _ = torch.max(ins_prediction, 0)  # Take max across instances
        else:
            max_prediction, _ = torch.max(ins_prediction, 1)

        max_loss = self.criterion(max_prediction.view(1, -1), bag_label.view(1, -1))

        # Weighted sum of the losses
        loss = self.args.single_weight_parameter * bag_loss + (1 - self.args.single_weight_parameter) * max_loss
        return loss

    def _get_optimizer(self) -> optim.Optimizer:
        """
        Set up the optimizer.
        """
        optimizer_cls = optim.AdamW if self.args.optimizer == "adamw" else optim.Adam
        return optimizer_cls(
            params=self.milnet.parameters(),
            lr=self.args.lr,
            betas=(self.args.betas[0], self.args.betas[1]),
            weight_decay=self.args.weight_decay
        )

    def _get_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Setup the learning rate scheduler.
        """
        if self.args.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=self.args.num_epochs,
                eta_min=self.args.eta_min
            )
        elif self.args.scheduler == 'cosinewarmup':
            return CosineWarmupScheduler(
                optimizer=self.optimizer,
                warmup_epochs=int(self.args.num_epochs / 20),
                max_epochs=self.args.num_epochs
            )
        else:
            print(f'Scheduler set to None')
            return None

    def train_step(self, x, bag_label: torch.Tensor) -> torch.Tensor:
        """
        Perform a single step of training.
        """
        # Forward pass through the model
        ins_prediction, bag_prediction, attentions = self.forward(x)

        # Compute loss
        loss = self.compute_loss(bag_prediction, bag_label, ins_prediction)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self, data, cur_epoch):
        self.milnet.train()
        total_loss = 0

        # Iterate through the data
        for i, (bag_label, bag_feats) in enumerate(data):
            # Forward pass and loss computation
            loss = self.train_step(bag_feats, bag_label)
            total_loss += loss.item()

        # Step the scheduler after each epoch
        if self.scheduler:
            self.scheduler.step()

        return total_loss / len(data)

    def save_model(self, save_path: str):
        """
        Save the model weights to the specified file.
        """
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, model_path: str):
        """
        Load the model weights from the specified file.
        """
        self.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")

# Example code to initialize and test
if __name__ == "__main__":
    class Args:
        feats_size = 768
        num_classes = 2  # Example for binary classification
        num_heads = 8
        mlp_multiplier = 4
        activation = 'relu'
        encoder_dropout = 0.1
        big_lambda = 0.5
        random_patch_share = 0.3
        depth = 4
        weight_init__weight_init_i__weight_init_b = ['xavier_normal', 'xavier_normal']
        single_weight_parameter = 0.5  # Set a value for single weight parameter
        scheduler = 'cosine'  # Specify the scheduler type
        lr = 0.0002  # Learning rate
        betas = [0.5, 0.9]  # AdamW betas
        weight_decay = 5e-3
        num_epochs = 200
        eta_min = 1e-6

    args = Args()

    # Initialize model
    model = Snuffy(args)
    

    # Example input (batch of images)
    sample_input = torch.randn(16, 3, 224, 224)  # Batch of 16 images of size 224x224 with 3 channels
    sample_labels = torch.randint(0, 2, (16,))  # Random labels for binary classification

    # Run model and get predictions
    ins_pred, bag_pred, attentions = model(sample_input)

    # Example loss computation
    loss = model.compute_loss(bag_pred, sample_labels, ins_pred)
    print("Loss:", loss.item())
