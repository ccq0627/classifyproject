from typing import Tuple, List, Dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from torch import Tensor
from PIL import Image
import os
import os.path as osp

class Dataset(Dataset):
    def __init__(self, root: str, transform: transforms.Compose) -> None:
        self.root = root
        self.transform = transform
        self.image_paths, self.classes, self.class_to_idx = self._load_data(root)

    def _load_data(self, root: str):
        classes = sorted(os.listdir(root))
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        image_paths = list(Path(root).glob("*/*.jpg"))
        return image_paths, classes, class_to_idx
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        image_path = self.image_paths[idx]
        image_pil = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image_pil)
        label = self.class_to_idx[osp.basename(osp.dirname(image_path))]
        return image, label


def create_dataloader(
        train_dir: str,
        test_dir: str,
        transform: Dict[str, transforms.Compose],
        batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    
    train_dataset = Dataset(root=train_dir, transform=transform["train"])
    test_dataset = Dataset(root=test_dir, transform=transform["test"])

    classes = train_dataset.classes

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return  train_loader, test_loader, classes

