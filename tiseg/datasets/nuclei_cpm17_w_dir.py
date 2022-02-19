from .builder import DATASETS
from .nuclei_custom_w_dir import NucleiCustomDatasetWithDirection


@DATASETS.register_module()
class NucleiCPM17DatasetWithDirection(NucleiCustomDatasetWithDirection):
    """CPM17 Nuclei segmentation dataset."""

    CLASSES = ('background', 'nuclei')

    PALETTE = [[0, 0, 0], [255, 2, 255]]

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', sem_suffix='_semantic.png', inst_suffix='_instance.npy', **kwargs)
