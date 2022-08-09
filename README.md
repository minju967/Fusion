# Fusion\n
combine Image features with mesh information.

conda activate FusionNet
    python 3.7
    pyvista

Model
    models.py
        Class MVCNN
            1) Multi-view images are extracted features by CNN model.
        Class MLP
            2) Mesh information(13 elements) is extracted features by MLP.
        Class Fusion
            3) combine 1) with 2).

Tools
    dataset.py
        class FusionDataset
            1) call all images
            2) call all objs
            3) create mesh information
            4) return (obj name, target, images, mesh 13 elements)
            5) number of class: 2 (0: positive, 1: negative)
                - positive: A, C, E
                - negative: B, D
