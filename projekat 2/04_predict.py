import sys
sys.path.insert(0, r"D:\Python_packages")

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

if __name__ == '__main__':
    os.environ["nnUNet_raw"] = r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\data\nnunet"
    os.environ["nnUNet_preprocessed"] = r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\data\preprocessed"
    os.environ["nnUNet_results"] = r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\results"

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=False,  
        device=torch.device('cuda'),
        verbose=True
    )

    predictor.initialize_from_trained_model_folder(
    r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\results\Dataset001_Mandible\nnUNetTrainer__nnUNetPlans__3d_lowres",
    use_folds=(0,),
    checkpoint_name="checkpoint_final.pth"   # <-- puno ime, sa ekstenzijom
    )

    predictor.predict_from_files(
        r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\data\nnunet\Dataset001_Mandible\imagesTs",
        r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\results\predictions",
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1
    )

    print("Predikcija zavrsena!")