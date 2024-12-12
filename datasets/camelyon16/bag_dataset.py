import os
import torch
import torchvision.transforms.functional as VF
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import argparse
from typing import List, Dict, Tuple
import timm
import yaml
import time 
 
class BagDataset:
    def __init__(self, files_list: List[str], transform=None, patch_labels_dict: Dict[str, int] = None):
        if patch_labels_dict is None:
            patch_labels_dict = {}
        self.files_list = files_list
        self.transform = transform
        self.patch_labels = patch_labels_dict

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)

        patch_address = os.path.join(
            *temp_path.split(os.path.sep)[-3:]  # class_name/bag_name/patch_name.jpeg
        )
        label = self.patch_labels.get(patch_address, -1)  # TCGA doesn't have patch labels, set -1 to ignore

        patch_name = Path(temp_path).stem
        # Camelyon16 Patch Name Convention: {row}_{col}-17.jpeg > 116_228-17.jpeg
        # TCGA       Patch Name Convention: {row}_{col}.jpeg    > 116_228-17.jpeg
        row, col = patch_name.split('-')[0].split('_')
        position = np.asarray([int(row), int(col)])

        sample = {
            'input': img,
            'label': label,
            'position': position
        }

        if self.transform:
            sample = self.transform(sample)
        return sample
    


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['input']
        img = VF.resize(img, self.size)
        return {
            **sample,
            'input': img
        }


class NormalizeImage:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['input']
        img = VF.normalize(img, self.mean, self.std)
        return {
            **sample,
            'input': img
        }


class ToTensor:
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)

        label = sample['label']
        assert isinstance(label, int), f"A sample label should be of type int, but {type(label)} received."
        return {
            **sample,
            'label': torch.tensor(label),
            'input': img
        }


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

# Define your VIT feature extractor
class VITFeatureExtractor(nn.Module):
    def __init__(self, base_model='vit_base_patch16_224', out_dim=768, pretrained=True):
        super(VITFeatureExtractor, self).__init__()
        self.model = timm.create_model(base_model, pretrained=pretrained, num_classes=0)
        
        num_ftrs = self.model.embed_dim
        print("Feature size (num_ftrs):", num_ftrs)
        
        # Projection MLP (optional)
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)
        
    def forward(self, x):
        # Extract features using the forward_features method of the ViT model
        fts = self.model.forward_features(x)  # Returns feature embeddings
        h = fts[:, 0, :]  # Extracts the [CLS] token's feature vector
        x = self.l1(h)
        x = torch.relu(x)
        x = self.l2(x)
        return h, x  
    
def bag_dataset(args, patches: List[str] = None, patch_labels_dict: dict = None) -> Tuple[DataLoader, int]:

    """
    Create a bag dataset and its corresponding data loader.

    This function creates a bag dataset from the provided list of patch file paths and prepares a data loader to access
    the data in batches. The bag dataset is expected to contain bag-level data, where each bag is represented as a
    collection of instances.

    Args:
        args (object): An object containing arguments or configurations for the data loader setup.
        patches (List[str]): A list of file paths representing patches.
        patch_labels_dict (dict): A dict in the form {patch_name: patch_label}

    Returns:
        tuple: A tuple containing two elements:
            - dataloader (torch.utils.data.DataLoader): The data loader to access the bag dataset in batches.
            - dataset_size (int): The total number of bags (patches) in the dataset.
    """
    if args.backbone in specified_archs:
        if args.transform == 1:
            transforms = [Resize(224), ToTensor(), NormalizeImage((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        else:
            transforms = [Resize(224), ToTensor()]
        transformed_dataset = BagDataset(
            files_list=patches,
            transform=Compose(transforms),
            patch_labels_dict=patch_labels_dict
        )
    else:
        transforms = [ToTensor()]
        if args.backbone == 'vitbasetimm':
            if args.transform == 1:
                transforms = [
                    Resize(224), 
                    ToTensor(), 
                    NormalizeImage((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ]
            else:
                transforms = [Resize(224), ToTensor()]
        transformed_dataset = BagDataset(
            files_list=patches,
            transform=Compose(transforms),
            patch_labels_dict=patch_labels_dict
        )
    dataloader = DataLoader(
        transformed_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        drop_last=False)
    
    return dataloader, len(transformed_dataset)


def get_patch_labels_dict(tile_label_csv) -> Optional[Dict[str, int]]:
    
    patch_labels_path = os.path.join('tile_label.csv')

    try:
        labels_df = pd.read_csv(patch_labels_path)
        print("- content of tile_label: ")
        print(labels_df.head(3))
        print(f'Using patch_labels csv file at {patch_labels_path}')
        duplicates = labels_df['slide_name'].duplicated()
        assert not any(duplicates), "There are duplicate patch_names in the {patch_labels_csv} file."
        return labels_df.set_index('slide_name')['label'].to_dict()

    except FileNotFoundError:
        print(f'No patch_labels csv file at {patch_labels_path}')
        return None 
    
def compute_feats(
        bags_list: List[str] = None,
        embedder: nn.Module = None,
        save_path: str = None,
        patch_labels_dict: dict = None, 
        csv_directory: str = None
): 
    print('embedder:', embedder)
    num_bags = len(bags_list)
    
    for i in tqdm(range(num_bags)):
        start_time = time.time()
        
        patches = glob.glob(os.path.join(bags_list[i], '*.jpg')) + \
                  glob.glob(os.path.join(bags_list[i], '*.jpeg'))

        dataloader, bag_size = bag_dataset(args, patches, patch_labels_dict)

        feats_list = []
        feats_labels = []
        feats_positions = []
        embedder.eval()
        
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().to(device)
                feats, classes = embedder(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
                batch_labels = batch['label']
                feats_labels.extend(np.atleast_1d(batch_labels.squeeze().tolist()).tolist())
                feats_positions.extend(batch['position'])

                tqdm.write(
                    '\r Computed: {}/{} -- {}/{}'.format(i + 1, num_bags, iteration + 1, len(dataloader)), end=''
                )

        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            df = pd.DataFrame(feats_list, dtype=np.float32)
            if args.dataset == 'camelyon16':
                df['label'] = feats_labels if patch_labels_dict is not None else np.nan
                df['position'] = feats_positions if patch_labels_dict is not None else None

            class_name, bag_name = bags_list[i].split(os.path.sep)[-2:]
            
            # csv_directory = os.path.join(save_path, class_name)
            csv_file = os.path.join(csv_directory, bag_name)

            os.makedirs(csv_directory, exist_ok=True)
            df_save_path = os.path.join(csv_file + '.csv')
            df.to_csv(df_save_path, index=False, float_format='%.4f')
        print(f"Take {time.time()-start_time} to process 1 batch") 
        
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()], add_help=False)
    args = parser.parse_args()
    args.slides_dir = config['SLIDES_DIR'] 
    
    args.slides_dir = config['SLIDES_DIR']
    args.tile_label_csv = config['TILE_LABEL_CSV'] 
    
    args.batch_size = 32 
    args.transform = 1 
    args.backbone = 'vitbasetimm' 
    args.num_workers = 1  
    gpu_index = 0 
     
    gpu_ids = tuple(args.gpu_index)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    # Get backbone and feature size
    embedder = VITFeatureExtractor()

    # Get the path for bags (patches)
    bag_path = '/project/hnguyen2/mvu9/camelyon16/single/single/normal'
    # patches = glob.glob(os.path.join(test_path, '*.jpeg'))
    

    feats_path = '/project/hnguyen2/mvu9/camelyon16/features/normal' 
    
    if os.path.exists(feats_path):
        shutil.rmtree(feats_path)
        print(f"Directory {feats_path} already existed and has been removed.")
    os.mkdir(feats_path)
       
    bags_list = glob.glob(os.path.join(bags_path))
    
    print(f'Number of bags: {len(bags_list)} | Sample Bag: {bags_list[0]}')

    # Get patch labels (simulated here)
    patch_labels_dict = get_patch_labels_dict(args.tile_label_csv)

    start_time = time.time()
    csv_directory = '/project/hnguyen2/mvu9/camelyon16/features' 
    
    compute_feats(
        args, 
        bags_list=bags_list, 
        embedder=embedder, 
        patch_labels_dict=patch_labels_dict, 
        csv_directory=csv_directory)

    print(f'Took {time.time() - start_time} seconds to compute feats')

    # Save the features (optional step)
    # save_class_features(args, feats_path)
 
    
 