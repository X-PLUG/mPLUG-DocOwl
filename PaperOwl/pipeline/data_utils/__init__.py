from .processors.builder import build_processors
from .xgpt3_dataset import MultiModalDataset

def train_valid_test_datasets_provider(data_path, config, tokenizer, seq_length=1024,image_root='ureader_images'):
    """Build train and valid datasets."""
    print('> building train and validation datasets for mPLUG-Owl ...')
    train_ds, valid_ds = build_train_valid_test_datasets(
        input_file=data_path,  
        tokenizer=tokenizer,
        max_length=seq_length, 
        config=config,
        image_root=image_root)
    print("> finished creating mPLUG-Owl datasets ...")

    return train_ds, valid_ds

def build_train_valid_test_datasets(input_file, tokenizer, max_length=80, config=None,image_root='ureader_images'):
    train_processors = build_processors(config['train_processors'])
    valid_processors = build_processors(config['valid_processors'])
    if isinstance(input_file, dict):
        train_ds = MultiModalDataset(input_file['train'][0], tokenizer, train_processors, max_length, image_root=image_root)
        valid_ds = {name: MultiModalDataset(ds, tokenizer, valid_processors, max_length) for name,ds in input_file['valid'].items()}
        test_ds = None

    else:
        assert len(input_file) == 2 # If you have files more than 2, modify code at here or merger them into train and dev
        train_ds = MultiModalDataset(input_file[0], tokenizer, train_processors, max_length)
        valid_ds = MultiModalDataset(input_file[1], tokenizer, valid_processors, max_length)
        test_ds = None
    return (train_ds, valid_ds)
