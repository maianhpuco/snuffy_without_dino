import argparse
import shutil
import pandas as pd
import numpy as np
import os
import ast
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as VF
import torchvision.models as models
import timm
import yaml
import time
import glob
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from lightly.utils.scheduler import CosineWarmupScheduler
from torch.autograd import Variable

from custom_layers import (
    MultiHeadedAttention, PositionwiseFeedForward, 
    BClassifier, FCLayer, MILNet, 
    VITFeatureExtractor, Encoder, EncoderLayer
)

class Snuffy(nn.Module):
    def __init__(self, args):
        super(Snuffy, self).__init__()
        self.args = args
        
        self.vit_extractor = VITFeatureExtractor(
            base_model='vit_base_patch16_224', out_dim=768, pretrained=True)
         
        self.milnet = self._get_milnet()  # Get the MILNet for instance and bag classification
        self.milnet.to(args.device) 
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

    def _get_milnet(self) -> nn.Module:
        """
        Creates the MILNet by combining the instance classifier and bag classifier.
        """
        
        i_classifier = FCLayer(
            in_size=self.args.feats_size, out_size=self.args.num_classes
            ).to(args.device)
        c = copy.deepcopy
        attn = MultiHeadedAttention(
            self.args.num_heads, self.args.feats_size
            ).to(args.device)
        ff = PositionwiseFeedForward(self.args.feats_size, self.args.feats_size * self.args.mlp_multiplier,
                                     self.args.activation, self.args.encoder_dropout).to(args.device)

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
        ).to(args.device)

        milnet = MILNet(
            i_classifier, 
            b_classifier
            ).to(args.device)

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
        
def get_patch_labels_dict(patch_labels_path) -> Optional[Dict[str, int]]:
    try:
        labels_df = pd.read_csv(patch_labels_path)
        ignore_values = ['slide_name', 'label']
        labels_df = labels_df[~labels_df.isin(ignore_values).any(axis=1)]
        labels_df = labels_df.drop_duplicates()
        print("- content of tile_label: ")
        print(f'Using patch_labels csv file at {patch_labels_path}')
        duplicates = labels_df['slide_name'].duplicated()
        assert not any(duplicates), "There are duplicate patch_names in the {patch_labels_csv} file."
        return labels_df.set_index('slide_name')['label'].to_dict()

    except FileNotFoundError:
        print(f'No patch_labels csv file at {patch_labels_path}')
        return None        
        
def get_bag_list(train_val_test, args):
    # Get the path for bags (patches)
    bags_path_normal = '/project/hnguyen2/mvu9/camelyon16/single/single/normal'
    bags_path_tumor  = '/project/hnguyen2/mvu9/camelyon16/single/single/tumor' 
    
    feats_path = '/project/hnguyen2/mvu9/camelyon16/features/normal' 
    
    # if os.path.exists(feats_path):
    #     shutil.rmtree(feats_path)
    #     print(f"Directory {feats_path} already existed and has been removed.")
    # os.mkdir(feats_path)

    split_df = pd.read_csv(args.sampling_csv)
    train_bags_list = split_df[train_val_test].dropna().tolist()  
    available_bags_list = glob.glob(os.path.join(bags_path_normal, '*')) + glob.glob(os.path.join(bags_path_tumor,'*')) 
    available_bags_base_names = [os.path.basename(bag) for bag in available_bags_list]
 
    filtered_bags_list = [bag for bag, base_name in zip(available_bags_list, available_bags_base_names) 
                    if base_name in train_bags_list]   
    bags_list = filtered_bags_list 
    
    return bags_list

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
 
# Example code to initialize and test
if __name__ == "__main__":
    config = load_config("./configs/compute_feats.yaml") 
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
        weight_init__weight_init_i__weight_init_b = ['xavier_normal', 'xavier_normal', 'xavier_normal']
        single_weight_parameter = 0.5  # Set a value for single weight parameter
        scheduler = 'cosine'  # Specify the scheduler type
        lr = 0.0002  # Learning rate
        betas = [0.5, 0.9]  # AdamW betas
        weight_decay = 5e-3
        num_epochs = 200
        eta_min = 1e-6
        optimizer = 'adam'
        device = "cuda" if torch.cuda.is_available() else "cpu" 
        l2normed_embeddings = 1 
        sampling_csv = './datasets/camelyon16/sample_20.csv'
        
    args = Args()
    
    vit_extractor = VITFeatureExtractor()
    model = Snuffy(args) 
    
    args.slides_dir = config['SLIDES_DIR'] 
    args.sampling_csv = config['SAMPLING_CSV']
    args.tile_label_csv = config['TILE_LABEL_CSV']  
    
    
    
    train_val_test = 'train'
    bags_list = get_bag_list(train_val_test, args) 
    num_bags = len(bags_list)
    print(f'>>>> Running {train_val_test} | Number of bags: {len(bags_list)} | Sample Bag: {bags_list[0]}') 
    
    patch_labels_dict = get_patch_labels_dict(args.tile_label_csv) 
    csv_directory = '/project/hnguyen2/mvu9/camelyon16/features'  
    
    vit_extractor = vit_extractor.to(args.device)
    model = model.to(args.device)
      
    for i in tqdm(range(num_bags)):
        start_time = time.time()
        
        patches = glob.glob(os.path.join(bags_list[i], '*.jpg')) + \
                  glob.glob(os.path.join(bags_list[i], '*.jpeg'))

        dataloader, bag_size = bag_dataset(args, patches, patch_labels_dict) 
        
        feats_list = torch.empty(0, feats_size, dtype=torch.float32, device=feats.device)
        feats_labels = torch.empty(0, dtype=torch.float32, device=args.device)  # labels as tensor
        feats_positions = torch.empty(0, dtype=torch.float32, device=args.device) 
        
        for iteration, batch in enumerate(dataloader):
            patches = batch['input'].float().to(args.device)
            _, feats = vit_extractor(patches)
            
            feats_list = torch.cat((feats_list, feats.cpu()), dim=0)
            
            feats_list.extend(feats)
            batch_labels = batch['label']           
            # feats_labels.extend(np.atleast_1d(batch_labels.squeeze().tolist()).tolist())
            feats_labels = torch.cat((feats_labels, batch_labels.squeeze().float().to(args.device)), dim=0)
 
            # feats_positions.extend(batch['position'])
            feats_positions = torch.cat((feats_positions, batch['position'].float().to(args.device)), dim=0)
            tqdm.write(
                '\r Computed: {}/{} -- {}/{}'.format(i + 1, num_bags, iteration + 1, len(dataloader)), end=''
            )
            print("each epochs-----")
            print(feats_list.shape)
            print(feats_labels.shape)
            print(feats_positions.shape)
            
        print("-----check_shape")
        print(feats_list.shape)
        print(feats_labels.shape)
        print(feats_positions.shape)
    
        # if self.args.l2normed_embeddings == 1:
        #     bag_feats = bag_feats / np.linalg.norm(bag_feats, axis=1, keepdims=True) 
            
        # bag_label = Variable(Tensor(np.array([bag_label])).to(device))
        # bag_feats = Variable(Tensor(np.array([bag_feats])).to(device)) 
        


# label, feats, feats_labels, positions 
# def _load_data(bags_df, args):
#     all_labels = []
#     all_feats = []
#     all_feats_labels = []
#     all_positions = []
#     all_slide_names = []

#     feats_labels_available = True

#     for i in tqdm(range(len(bags_df))):
#         label, feats, feats_labels, positions = get_bag_feats(bags_df.iloc[i], args)
#         all_labels.append(label)
#         all_feats.append(feats)

#         if feats_labels is None:
#             feats_labels_available = False
#         if feats_labels_available:
#             all_feats_labels.append(feats_labels)
#             all_positions.append(positions)
#         all_slide_names.append(bags_df.iloc[i]['0'].split('/')[-1].split('.')[0])

#     if not feats_labels_available:
#         all_feats_labels = None
#         all_positions = None

#     return all_labels, all_feats, all_feats_labels, all_positions, all_slide_names
 
 
# label, feats, feats_labels, positions = get_bag_feats(row, args)
#     slide_name = row['0'].split('/')[-1].split('.')[0]
#     return label, feats, feats_labels, positions, slide_name 


    # # Initialize model
    # 
    

    # # Example input (batch of images)
    # sample_input = torch.randn(50, 3, 224, 224)  # Batch of 16 images of size 224x224 with 3 channels
    # sample_labels = torch.randint(0, 2, (16,))  # Random labels for binary classification
    
    # sample_input = sample_input.to(args.device)  # Move input to the device (GPU or CPU)
    # sample_labels = sample_labels.to(args.device)  # Move labels to the device
 
    # # Run model and get predictions
    # ins_pred, bag_pred, attentions = model(sample_input)

    # # Example loss computation
    # loss = model.compute_loss(bag_pred, sample_labels, ins_pred)
    # print("Loss:", loss.item())
