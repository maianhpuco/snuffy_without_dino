from dataset.camelyon


specified_archs = [
    'vit_small', 'vit_base',
    'mae_vit_base_patch16', 'mae_vit_large_patch16'
] 

def compute_feats(
        args,
        bags_list: List[str],
        embedder: nn.Module,
        save_path: str,
        patch_labels_dict: dict = None
):
    print('embedder:', embedder)
    num_bags = len(bags_list)
    for i in tqdm(range(num_bags)):
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

            split_name, class_name, bag_name = bags_list[i].split(os.path.sep)[-3:]
            csv_directory = os.path.join(save_path, split_name, class_name)
            csv_file = os.path.join(csv_directory, bag_name)
            os.makedirs(csv_directory, exist_ok=True)
            df_save_path = os.path.join(csv_file + '.csv')
            df.to_csv(df_save_path, index=False, float_format='%.4f') 


def get_bags_path(args):
    '''
    example: /project/hnguyen2/mvu9/camelyon16/single/single/normal/normal_122 
    '''
    bags_path = os.path.join(
        DATASETS_PATH, args.dataset, 'single',
        args.fold,
        '*',  # train/test/val
        '*',  # classes: 0_normal 1_tumor
        '*',  # bag name
    )

    return bags_path 

def get_patch_labels_dict(args) -> Optional[Dict[str, int]]:
    patch_labels_path = os.path.join(DATASETS_PATH, args.dataset, 'tile_label.csv')

    try:
        labels_df = pd.read_csv(patch_labels_path)
        print(f'Using patch_labels csv file at {patch_labels_path}')
        duplicates = labels_df['slide_name'].duplicated()
        assert not any(duplicates), "There are duplicate patch_names in the {patch_labels_csv} file."
        return labels_df.set_index('slide_name')['label'].to_dict()

    except FileNotFoundError:
        print(f'No patch_labels csv file at {patch_labels_path}')
        return None

def main():
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    backbone, num_feats = get_embedder_backbone(args)

    #get bag  paths 
    bags_path = get_bags_path(args)
    print(f'Using bags at {bags_path}')
    
    #create 
    if 'Supervised' in args.embedder:
        feats_path = os.path.join(EMBEDDINGS_PATH, args.dataset, args.embedder)
    else:
        feats_path = os.path.join(EMBEDDINGS_PATH, args.dataset, args.embedder + "_" + args.version_name)

    os.makedirs(feats_path, exist_ok=True)
    
    bags_list = glob.glob(bags_path)
    
    print(f'Number of bags: {len(bags_list)} | Sample Bag: {bags_list[0]}')

    patch_labels_dict = get_patch_labels_dict(args)

    start_time = time.time()
    embedder, _ = get_embedder(args, backbone, num_feats)
    compute_feats(args, bags_list, embedder, feats_path, patch_labels_dict)

    print(f'Took {time.time() - start_time} seconds to compute feats')
    save_class_features(args, feats_path)

if __name__ == '__main__': 

#TODO:
# bag_path 
# feature_path 
# embedders 
# compute feature by training 