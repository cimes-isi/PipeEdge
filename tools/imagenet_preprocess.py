"""Process the ImageNet tarballs into a usable directory structure."""
import argparse
from torchvision.datasets import ImageNet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ImageNet tarballs")
    parser.add_argument("root", type=str,
                        help="The root directory containing tarballs: " +
                             "ILSVRC2012_img_train.tar, ILSVRC2012_img_val.tar, and " +
                             "ILSVRC2012_devkit_t12.tar.gz")
    args = parser.parse_args()

    for split in ['train', 'val']:
        # NOTE: ImageNet class depends on scipy, which isn't an automatic torchvision dependency
        print(f"Initializing split: {split}")
        dataset = ImageNet(args.root, split=split)
