import os

import nibabel
import pandas as pd
import numpy as np

from nilearn import datasets
from nilearn.image import resample_to_img

from nistats.second_level_model import SecondLevelModel
from nistats.thresholding import map_threshold


def make_zmaps(model, contrasts):
    """ Given a model and contrasts, return the corresponding z-maps"""
    z_maps = {}
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        z_maps[contrast_id] = model.compute_contrast(
            contrast_val, output_type='z_score')
    return z_maps

def prepare_data(oasis_dataset):
    gray_matter_map_filenames = oasis_dataset.gray_matter_maps
    # Get a mask image: A mask of the  cortex of the ICBM template
    gm_mask = datasets.fetch_icbm152_brain_gm_mask()
    # Resample the images, since this mask has a different resolution
    mask_img = resample_to_img(
        gm_mask, gray_matter_map_filenames[0], interpolation='nearest')
    return mask_img, gray_matter_map_filenames


def make_stats_model(mask_img, gray_matter_map_filenames, design_matrix):
    # Specify and fit the second-level model when loading the data, we
    # smooth a little bit to improve statistical behavior
    second_level_model = SecondLevelModel(smoothing_fwhm=2.0, mask=mask_img)
    second_level_model.fit(gray_matter_map_filenames,
                           design_matrix=design_matrix)
    # Estimate the contrast is very simple. We can just provide the column
    # name of the design matrix.
    z_map = second_level_model.compute_contrast(second_level_contrast=[1, 0, 0],
                                                output_type='z_score')
    # We threshold the second level contrast at uncorrected p < 0.001 and plot it.
    # First compute the threshold.
    _, threshold = map_threshold(
        z_map, alpha=.05, height_control='fdr')
    # Can also study the effect of sex: compute the stat, compute the
    # threshold, plot the map
    z_map = second_level_model.compute_contrast(second_level_contrast='sex',
                                                output_type='z_score')
    _, threshold = map_threshold(
        z_map, alpha=.05, height_control='fdr')
    z_maps = {'age': z_map}
    return second_level_model, z_maps
    
    
def make_design_matrix(oasis_dataset, n_subjects):
    age = oasis_dataset.ext_vars['age'].astype(float)
    sex = oasis_dataset.ext_vars['mf'] == b'F'
    intercept = np.ones(n_subjects)
    design_matrix = pd.DataFrame(np.vstack((age, sex, intercept)).T,
                                 columns=['age', 'sex', 'intercept'])
    design_matrix = pd.DataFrame(design_matrix, columns=['age', 'sex',
                                                         'intercept']
                                 )
    return design_matrix


def run_reporter(model, mask_img, design_matrix, contrast, z_maps):
    icbm152_2009 = datasets.fetch_icbm152_2009()
    output_filepath = 'generated_report_slm_oasis.html'
    from nistats.reporting.glm_reporter import generate_report
    generate_report(
            output_filepath,
            model,
            contrasts=contrast,
            z_maps=z_maps,
            mask=mask_img,
            design_matrices=[design_matrix],
            bg_img=icbm152_2009['t1'],
            threshold=3.09,
            display_mode='z',
            )
    # generate_subject_stats_report(
    #         output_filepath,
    #         contrasts=contrast,
    #         z_maps=z_maps,
    #         mask=mask_img,
    #         design_matrices=[design_matrix],
    #         anat=icbm152_2009['t1'],
    #         )


def get_zmap(mask_img, gray_matter_map_filenames, design_matrix, contrast):
    zmap_filepath = os.path.join(os.path.dirname(__file__), 'oasis_zmap.nii.gz')
    second_level_model, z_maps = make_stats_model(mask_img,
                                                  gray_matter_map_filenames,
                                                  design_matrix,
                                                  )
    z_maps = make_zmaps(second_level_model, contrast)
    z_maps['age'].to_filename(zmap_filepath)
    return second_level_model, z_maps


def create_report_oasis():
    n_subjects = 100  # more subjects requires more memory
    contrast = {'age': [1, 0, 0], 'sex':[0, 1, 0]}
    oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
    mask_img, gray_matter_map_filenames = prepare_data(oasis_dataset)
    design_matrix = make_design_matrix(oasis_dataset, n_subjects)
    model, z_maps = get_zmap(mask_img,
                      gray_matter_map_filenames,
                      design_matrix,
                      contrast,
                      )
    model.design_matrices_ = [model.design_matrix_]
    run_reporter(model, mask_img, design_matrix, contrast, z_maps)
    
    
if __name__ == '__main__':
    create_report_oasis()
