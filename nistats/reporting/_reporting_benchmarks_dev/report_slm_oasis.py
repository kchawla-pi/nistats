import pandas as pd
import numpy as np

from nilearn import datasets
from nilearn.image import resample_to_img

from nistats.reporting.glm_reporter import make_glm_report
from nistats.second_level_model import SecondLevelModel


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


def make_report_oasis():
    n_subjects = 5  # more subjects requires more memory
    oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
    # Resample the images, since this mask has a different resolution
    mask_img = resample_to_img(datasets.fetch_icbm152_brain_gm_mask(),
                               oasis_dataset.gray_matter_maps[0],
                               interpolation='nearest',
                               )
    design_matrix = make_design_matrix(oasis_dataset, n_subjects)
    second_level_model = SecondLevelModel(smoothing_fwhm=2.0, mask=mask_img)
    second_level_model.fit(oasis_dataset.gray_matter_maps,
                           design_matrix=design_matrix)

    contrast = [[1, 0, 0], [0, 1, 0]]
    report = make_glm_report(
            model=second_level_model,
            roi_img=mask_img,
            contrasts=contrast,
            bg_img=datasets.fetch_icbm152_2009()['t1'],
            height_control=None,
            )
    output_filepath = 'generated_report_slm_oasis.html'
    report.save_as_html(output_filepath)


if __name__ == '__main__':
    make_report_oasis()
