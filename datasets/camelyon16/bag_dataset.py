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