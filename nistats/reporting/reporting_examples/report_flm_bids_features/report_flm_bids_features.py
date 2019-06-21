import os

import nibabel as nib
import numpy as np
from patsy import DesignInfo
from nilearn.datasets import load_mni152_template

from nistats.reporting.glm_reporter import make_glm_report
from nistats.utils import get_design_from_fslmat
from nistats.datasets import (fetch_openneuro_dataset_index,
                              fetch_openneuro_dataset, select_from_index)

def fetch_bids_data():
    _, urls = fetch_openneuro_dataset_index()
    
    exclusion_patterns = ['*group*', '*phenotype*', '*mriqc*',
                          '*parameter_plots*', '*physio_plots*',
                          '*space-fsaverage*', '*space-T1w*',
                          '*dwi*', '*beh*', '*task-bart*',
                          '*task-rest*', '*task-scap*', '*task-task*']
    urls = select_from_index(
        urls, exclusion_filters=exclusion_patterns, n_subjects=1)
    
    data_dir, _ = fetch_openneuro_dataset(urls=urls)
    return data_dir


def make_flm(data_dir):
    from nistats.first_level_model import first_level_models_from_bids
    task_label = 'stopsignal'
    space_label = 'MNI152NLin2009cAsym'
    derivatives_folder = 'derivatives/fmriprep'
    models, models_run_imgs, models_events, models_confounds = \
        first_level_models_from_bids(
            data_dir, task_label, space_label, smoothing_fwhm=5.0,
            derivatives_folder=derivatives_folder)

    model, imgs, events, confounds = (
        models[0], models_run_imgs[0], models_events[0], models_confounds[0])
    subject = 'sub-' + model.subject_label
    design_matrix = make_design_matrix(data_dir, subject)
    model.fit(imgs, design_matrices=[design_matrix])
    return model, subject


def make_design_matrix(data_dir, subject):
    fsl_design_matrix_path = os.path.join(
        data_dir, 'derivatives', 'task', subject, 'stopsignal.feat', 'design.mat')
    design_matrix = get_design_from_fslmat(
        fsl_design_matrix_path, column_names=None)

    design_columns = ['cond_%02d' % i for i in range(len(design_matrix.columns))]
    design_columns[0] = 'Go'
    design_columns[4] = 'StopSuccess'
    design_matrix.columns = design_columns
    return design_matrix


def create_report_bids_features():
    data_dir = fetch_bids_data()
    model, subject = make_flm(data_dir)
    output_filepath = 'generated_report_flm_bids_features.html'
    make_glm_report(output_path=output_filepath,
                    model=model,
                    contrasts='StopSuccess - Go',
                    )


if __name__ == '__main__':
    create_report_bids_features()
