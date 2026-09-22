"""Pheno4D dataset for Pointcept.

Copy to: Pointcept/pointcept/datasets/pheno4d.py
and add to  pointcept/datasets/__init__.py:
    from .pheno4d import Pheno4DDataset

The converted layout already matches DefaultDataset (coord/segment/instance npy
per scan folder), so this only fixes up the display name. `split` is the list of
PLANT folder names, which is how the plant-level split is enforced.
"""
import os
from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class Pheno4DDataset(DefaultDataset):
    """data_root/<Plant>/<scan>/{coord,segment,instance}.npy"""

    def get_data_name(self, idx):
        remain, scan = os.path.split(self.data_list[idx % len(self.data_list)])
        _, plant = os.path.split(remain)
        return f"{plant}-{scan}"
