specified_archs = [
    'vit_small', 'vit_base',
    'mae_vit_base_patch16', 'mae_vit_large_patch16'
] 


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