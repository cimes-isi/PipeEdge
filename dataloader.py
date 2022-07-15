"""DataLoader functions."""
import argparse
import os
from typing import Optional, Sequence, Union
import datasets
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DeiTFeatureExtractor, ViTFeatureExtractor
import model_cfg


def _get_ubatch_accuracy(output: torch.Tensor, target: torch.Tensor, topk_idx: int=0):
    # TODO: consistent interface - "target" should always either be a value (label) or index?
    # topk returns (values, indices) - which we need depends on the dataset and model
    with torch.no_grad():
        pred = output.topk(1)[topk_idx].t()
        batch_correct = pred.eq(target.view(1, -1).expand_as(pred)).sum().item()
    return batch_correct

class AccuracyTracker:
    """Track prediction accuracy."""

    def __init__(self) -> None:
        self._total_predicted = 0
        self._total_correct = 0

    def update(self, ubatch_size: int, ubatch_correct: int) -> None:
        """Recompute state for new predictions."""
        self._total_correct += ubatch_correct
        self._total_predicted += ubatch_size

    @property
    def count(self) -> int:
        """The total count."""
        return self._total_predicted

    @property
    def correct(self) -> int:
        """The correct count."""
        return self._total_correct

    @property
    def accuracy(self) -> float:
        """The current accuracy: `0 <= accuracy <= 1`."""
        return self._total_correct / self._total_predicted if self._total_predicted > 0 else 0.0


def get_imagefolder_dataset(model_name: str, root: str, split: str,
                            indices: Optional[Sequence]=None) -> datasets.Dataset:
    """Get an imagefolder dataset s.t. when it's iterated over, the data is in PyTorch format."""
    if model_name.startswith('facebook/deit'):
        feature_extractor = DeiTFeatureExtractor.from_pretrained(model_name)
    else:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    # We don't have to process microbatches like for text tokenization, which requires using 'map'.
    # A 'map' function requires a large up-front overhead to preprocess the entire dataset.
    # A 'transform' function runs on-the-fly and its overhead is not likely to be our bottleneck.
    def transform(item):
        """Transform item fields to Tensors."""
        # Pop to drop the the non-Tensor field(s) from the output.
        imgs = [img.convert('RGB') for img in item.pop('image')]
        feat = feature_extractor(images=imgs, return_tensors='pt')
        item['pixel_values'] = feat['pixel_values']
        item['label'] = torch.tensor(item['label'], dtype=torch.int32)
        return item
    data_files = { split: [os.path.join(root, split, "**")] }
    dataset = datasets.load_dataset('imagefolder', split=split, data_files=data_files)
    dataset.set_transform(transform)
    if indices is not None:
        dataset = dataset.select(indices)
    return dataset


def get_glue_dataset(model_name: str, config: str, split: str, ubatch_size: int,
                     indices: Optional[Sequence]=None, tok_padding: Union[bool, str]=True) \
    -> datasets.Dataset:
    """Create a GLUE dataset s.t. when it's iterated over, the data is in PyTorch format."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # When doing inference in batches (ubatch_size > 1), each item (tokenized sentence) in a batch
    # must have the same length, which requires padding shorter sentences in the batch.
    # 'transform' only operates on single items, so we'd be forced to use padding='max_length',
    # which always forces very long tensors, resulting in slower inference.
    # 'map' operates on batches, which allows for per-batch padding optimization.
    # 'transform' runs on-the-fly during dataset iteration, 'map' runs in advance and caches data.
    def map_function(batch):
        """Tokenize sentences in microbatches."""
        # Using return_tensors='pt' requires splitting the tensors afterward.
        # Use a numpy array instead, which will be stacked into a single PyTorch tensor later.
        encoding = tokenizer(batch['sentence'], padding=tok_padding, truncation=True,
                             return_tensors='np')
        batch.update(encoding)
        return batch
    dataset = datasets.load_dataset('glue', name=config, split=split)
    dataset = dataset.map(function=map_function, batched=True, batch_size=ubatch_size,
                          remove_columns=['sentence'])
    dataset.set_format(type='torch')
    if indices is not None:
        dataset = dataset.select(indices)
    return dataset


def main():
    """Main function."""
    # load dataset
    indices = None if args.batch_size is None else list(range(args.batch_size))
    if args.dataset == 'imagefolder':
        dataset_root = args.dataset_root
        if dataset_root is None:
            dataset_root = 'ImageNet'
            print(f"Dataset root not set, assuming: {dataset_root}")
        dataset = get_imagefolder_dataset(args.model_name, dataset_root, args.dataset_split,
                                          indices=indices)
        key_input, key_label = 'pixel_values', 'label'
        topk_idx = 1
    elif args.dataset == 'GLUE':
        dataset_config = args.dataset_config
        if dataset_config is None:
            dataset_config = 'cola'
            print(f"Dataset config not set, assuming: {dataset_config}")
        dataset = get_glue_dataset(args.model_name, dataset_config, args.dataset_split,
                                   args.ubatch_size, indices=indices)
        key_input, key_label = 'input_ids', 'label'
        topk_idx = 0
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    data_loader = DataLoader(dataset, batch_size=args.ubatch_size,
                             num_workers=args.dataloader_num_workers)

    # create module
    model_file = args.model_file
    if model_file is None:
        model_file = model_cfg.get_model_default_weights_file(args.model_name)
    layer_end = model_cfg.get_model_layers(args.model_name)
    model = model_cfg.module_shard_factory(args.model_name, model_file, 1, layer_end, 0)

    # run inference
    acc_tracker = AccuracyTracker()
    for data in data_loader:
        outputs = model(data[key_input])
        target = data[key_label]
        ubatch_correct = _get_ubatch_accuracy(outputs, target, topk_idx=topk_idx)
        acc_tracker.update(target.size(0), ubatch_correct)
        print(f"The accuracy so far is: {100*acc_tracker.accuracy:.2f}")
    print(f"Final Accuracy: {100*acc_tracker.accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Loader",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Model options
    parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224",
                        choices=model_cfg.get_model_names(),
                        help="the neural network model for loading")
    parser.add_argument("-M", "--model-file", type=str,
                        help="the model file, if not in working directory")
    # Input options
    parser.add_argument("-b", "--batch-size", type=int, help="batch size")
    parser.add_argument("-u", "--ubatch-size", default=8, type=int, help="microbatch size")
    # Data loader options
    # num_workers > 0 is slow to break the processing for loop after data [sub]set completes
    parser.add_argument("-dnw", "--dataloader-num-workers", default=0, type=int,
                        help="dataloader worker threads (0 uses the main thread)")
    # Dataset options
    dset = parser.add_argument_group('Dataset arguments')
    dset.add_argument("-ds", "--dataset", type=str, default="imagefolder", choices=['imagefolder', 'GLUE'],
                      help="dataset to use")
    dset.add_argument("-dss", "--dataset-split", default='train', type=str,
                      help="dataset split (depends on dataset), e.g.: train, val, validation, test")
    # NOTE: dataset=GLUE doesn't use --dataset-root
    dset.add_argument("-dsr", "--dataset-root", type=str,
                      help="dataset root directory (e.g., for ImageNet using dataset=imagefolder, "
                           "must contain 'ILSVRC2012_devkit_t12.tar.gz' and at least one of: "
                           "'ILSVRC2012_img_train.tar', 'ILSVRC2012_img_val.tar'")
    # NOTE: dataset=imagefolder doesn't use --dataset-config
    dset.add_argument("-dsc", "--dataset-config", type=str,
                      help="Dataset configuration name (e.g., for dataset=GLUE: 'cola' or 'sst2')")
    args = parser.parse_args()

    main()
