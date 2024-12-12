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


def bag_dataset(args, patches: List[str], patch_labels_dict: dict = None) -> Tuple[DataLoader, int]:
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
                transforms = [Resize(224), ToTensor(), NormalizeImage((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            else:
                transforms = [Resize(224), ToTensor()]
        transformed_dataset = BagDataset(
            files_list=patches,
            transform=Compose(transforms),
            patch_labels_dict=patch_labels_dict
        )
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)