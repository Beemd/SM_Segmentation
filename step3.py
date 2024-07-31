# Step 3: Load the trained model and make predictions on all generated png images
# Saves the generated files (.pkl, .npz, and .png) in a a new specified folder.
 
import os
import torch
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from batchgenerators.utilities.file_and_folder_operations import join

gpu_index = 0

# define the input and output directories
input_dir = '/share/dept_machinelearning/Faculty/Rasool, Ghulam/Shared Resources/Pancreatic Cancer Image Data/result_files/png_slices_L3_additional_unprocessed'

# instantiate the nnUNetPredictor
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=torch.device(type='cuda', index=gpu_index),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True
)
prob_sm_list = []
for j in range(1,3):
    # define the directory containing trained model weights
    dir = '/home/80024222/projects/nnUNet_results/Dataset50{}_NatGeSM_/nnUNetTrainer__nnUNetPlans__2d'.format(j)
    for i in range(0, 5):
        output_dir = '/share/dept_machinelearning/Faculty/Rasool, Ghulam/Shared Resources/Pancreatic Cancer Image Data/result_files/nn_unet_sm_additional_unprocessed/ensemble_{}_{}'.format(j,i)        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # initialize the network architecture and load the checkpoint
        predictor.initialize_from_trained_model_folder(dir,
        use_folds=(i,),
        checkpoint_name='checkpoint_final.pth',
        )

        # predict segmentations for all files in the input directory
        predictor.predict_from_files(
            input_dir,
            output_dir,
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )
