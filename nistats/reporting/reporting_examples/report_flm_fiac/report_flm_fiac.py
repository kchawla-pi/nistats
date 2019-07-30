import pandas as pd
import numpy as np

from nistats import datasets
from nistats.first_level_model import FirstLevelModel
from nistats.reporting import glm_reporter as glmr


def pad_vector(contrast_, n_columns):
    """A small routine to append zeros in contrast vectors"""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))


def test_flm_fiac_test():
    data = datasets.fetch_fiac_first_level()
    fmri_img = [data['func1'], data['func2']]
    
    from nilearn.image import mean_img
    mean_img_ = mean_img(fmri_img[0])
    
    design_files = [data['design_matrix1'], data['design_matrix2']]
    design_matrices = [pd.DataFrame(np.load(df)['X']) for df in design_files]
    
    fmri_glm = FirstLevelModel(mask_img=data['mask'], minimize_memory=True)
    fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrices)
    
    n_columns = design_matrices[0].shape[1]
    
    
    contrasts = {
        'SStSSp_minus_DStDSp': pad_vector([1, 0, 0, -1], n_columns),
        'DStDSp_minus_SStSSp': pad_vector([-1, 0, 0, 1], n_columns),
        'DSt_minus_SSt': pad_vector([-1, -1, 1, 1], n_columns),
        'DSp_minus_SSp': pad_vector([-1, 1, -1, 1], n_columns),
        'DSt_minus_SSt_for_DSp': pad_vector([0, -1, 0, 1], n_columns),
        'DSp_minus_SSp_for_DSt': pad_vector([0, 0, -1, 1], n_columns),
        'Deactivation': pad_vector([-1, -1, -1, -1, 4], n_columns),
        'Effects_of_interest': np.eye(n_columns)[:5]
                 }
    report = glmr.make_glm_report(fmri_glm, contrasts, bg_img=mean_img_,
                                  roi_img=data['mask'])
    output_filepath = 'generated_report_flm_fiac.html'
    report.save_as_html(output_filepath)

if __name__ == '__main__':
    test_flm_fiac_test()
