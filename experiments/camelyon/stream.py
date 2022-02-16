from typing import Tuple, Generator
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms


def stream_camelyon(
    split: str='train', 
    frac: float=1.0,
    root_dir: str='data',
    img_size: Tuple[int]=(96,96)
) -> Generator:
    """
    Split should be in ['train', 'val', 'test', 'id_val', 'id_test']
    """
    try:
        camelyon = get_dataset('camelyon17', root_dir=root_dir)
    except:
        camelyon = get_dataset('camelyon17', root_dir=root_dir, download=True)
    ds = camelyon.get_subset(
        split,
        frac,
        transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    )
    ds_iter = iter(get_train_loader('standard', ds, batch_size=1))

    while True:
        try:
            img = next(ds_iter)[0][0]
        except Exception:
            ds_iter = iter(get_train_loader('standard', ds, batch_size=1))
            img = next(ds_iter)[0][0]
        yield img