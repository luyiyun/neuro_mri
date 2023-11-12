import logging
from typing import Dict, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from monai.data import Dataset
from monai.transforms import (CenterSpatialCropd, Compose, EnsureChannelFirstd,
                              EnsureTyped, LoadImaged,
                              RandAdjustContrastd, RandBiasFieldd, RandFlipd,
                              Resized, ScaleIntensityd, Lambdad, adaptor)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

TUPLE3 = Tuple[int, int, int]


def get_transforms(
    train: bool = True,
    spatial_size: TUPLE3 = (256, 256, 24),
    center_crop: Optional[TUPLE3] = None,
    trunc_slice_top: Optional[int] = None,
    trunc_slice_bottom: Optional[int] = None,
    augmentations: Sequence[str] = ("randbias", "randcontrast"),
    classes: Optional[Sequence] = None,
):
    trans = [
        LoadImaged(keys=["img"]),
        EnsureChannelFirstd(keys=["img"]),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=spatial_size),
    ]
    if center_crop is not None:
        trans.append(CenterSpatialCropd(["img"], center_crop))
    if trunc_slice_bottom is not None or trunc_slice_top is not None:
        ss = slice(trunc_slice_bottom, -trunc_slice_top)
        trans.append(adaptor(lambda img: img[..., ss], "img"))
    if train:
        augments = {
            "randflip": RandFlipd(keys=["img"], prob=0.5, spatial_axis=0),
            "randbias": RandBiasFieldd(
                keys=["img"], prob=0.5, coeff_range=(0.2, 0.3)
            ),
            "randcontrast": RandAdjustContrastd(keys=["img"], prob=0.5),
        }
        for name in augmentations:
            trans.append(augments[name])
    if classes is not None:
        classes = list(classes)
        trans.extend(
            [
                Lambdad(keys="label", func=lambda x: classes.index(x)),
                EnsureTyped(
                    keys=["label"], data_type="tensor", dtype=torch.long
                ),
            ]
        )
    trans.append(
        EnsureTyped(
            keys=["img"],
            data_type="tensor",
            dtype=torch.float32,
            track_meta=False,
        )
    )
    return Compose(trans)


def get_loaders(
    df: pd.DataFrame,
    fn_col: str,
    label_col: str,
    valid_size: Optional[float] = None,
    test_size: Optional[float] = None,
    seed: Optional[int] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    drop_last: bool = True,
    return_classes_codes: bool = False
) -> Union[Dict[str, DataLoader], Tuple[Dict[str, DataLoader], Sequence]]:
    # get classes mapping
    classes = df[label_col].unique()
    logging.info(
        "classes mapping: %s"
        % ", ".join(["%s->%d" % (str(ci), i) for i, ci in enumerate(classes)])
    )

    # split dataframe
    dfs = {}
    if valid_size is not None:
        df, valid_df = train_test_split(
            df,
            test_size=valid_size,
            random_state=seed,
            shuffle=True,
            stratify=df[label_col],
        )
        dfs["valid"] = valid_df
    if test_size is not None:
        df, test_df = train_test_split(
            df,
            test_size=test_size
            if valid_size is None
            else test_size / (1 - valid_size),
            random_state=seed,
            shuffle=True,
            stratify=df[label_col],
        )
        dfs["test"] = test_df
    dfs["train"] = df

    # get datasets
    datasets = {}
    for phase, dfi in dfs.items():
        datasets[phase] = Dataset(
            data=[
                {"img": img, "label": label}
                for img, label in zip(
                    dfi[fn_col].values, dfi[label_col].values
                )
            ],
            transform=get_transforms(
                train=(phase == "train"), classes=classes
            ),
        )

    # get dataloaders
    dataloaders = {}
    for phase, datai in datasets.items():
        dataloaders[phase] = DataLoader(
            datai,
            batch_size=batch_size,
            shuffle=(phase == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last and (phase == "train"),
        )

    if not return_classes_codes:
        return dataloaders

    return dataloaders, classes
