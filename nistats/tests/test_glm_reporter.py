import os

import nibabel as nib
import numpy as np

from nose import SkipTest

import pandas as pd
from nilearn.datasets import fetch_oasis_vbm

from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import FirstLevelModel
from nistats.reporting import glm_reporter as glmr
from nistats import datasets
from numpy.testing import dec

from nistats.second_level_model import SecondLevelModel

try:
    import matplotlib as mpl
except ImportError:
    not_have_mpl = True
else:
    not_have_mpl = False


@dec.skipif(not_have_mpl)
def test_flm_fiac_test():
    if mpl.__version__ == '1.5.1':
        raise SkipTest('Skipping test in Matplotlib v1.5.1')
    data = datasets.fetch_fiac_first_level()
    fmri_img = [data['func1'], data['func2']]
    
    from nilearn.image import mean_img
    mean_img_ = mean_img(fmri_img[0])
    
    design_files = [data['design_matrix1'], data['design_matrix2']]
    design_matrices = [pd.DataFrame(np.load(df)['X']) for df in design_files]
    
    fmri_glm = FirstLevelModel(mask_img=data['mask'], minimize_memory=True)
    fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrices)
    
    n_columns = design_matrices[0].shape[1]
    
    def pad_vector(contrast_, n_columns):
        """A small routine to append zeros in contrast vectors"""
        return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))
    
    contrasts = {'SStSSp_minus_DStDSp': pad_vector([1, 0, 0, -1], n_columns),
                 'DStDSp_minus_SStSSp': pad_vector([-1, 0, 0, 1], n_columns),
                 }
    report_flm = glmr.make_glm_report(fmri_glm, contrasts, bg_img=mean_img_,
                                  roi_img=data['mask'])


def _make_design_matrix(oasis_dataset, n_subjects):
    age = oasis_dataset.ext_vars['age'].astype(float)
    sex = oasis_dataset.ext_vars['mf'] == b'F'
    intercept = np.ones(n_subjects)
    design_matrix = pd.DataFrame(np.vstack((age, sex, intercept)).T,
                                 columns=['age', 'sex', 'intercept'])
    design_matrix = pd.DataFrame(design_matrix, columns=['age', 'sex',
                                                         'intercept']
                                 )
    return design_matrix

@dec.skipif(not_have_mpl)
def test_slm_oasis_glass():
    n_subjects = 4
    contrast = [[1, 0, 0], [0, 1, 0]]
    oasis_dataset = fetch_oasis_vbm(n_subjects)
    design_matrix = _make_design_matrix(oasis_dataset, n_subjects)
    
    second_level_model = SecondLevelModel(smoothing_fwhm=2.0)
    second_level_model.fit(oasis_dataset.gray_matter_maps,
                           design_matrix=design_matrix)

    report_oasis = glmr.make_glm_report(
            model=second_level_model,
            contrasts=contrast,
            plot_type='glass',
            )


def test_make_contrasts_dict_with_string():
    test_input = 'StopSuccess - Go'
    expected_output = {'StopSuccess - Go': 'StopSuccess - Go'}
    actual_output = glmr._make_contrasts_dict(test_input)
    assert actual_output == expected_output


def test_make_contrasts_dict_with_list_of_strings():
    test_input = ['contrast_name_0', 'contrast_name_1']
    expected_output = {'contrast_name_0': 'contrast_name_0',
                       'contrast_name_1': 'contrast_name_1',
                       }
    actual_output = glmr._make_contrasts_dict(test_input)
    assert actual_output == expected_output


def test_make_contrasts_dict_with_dict():
    test_input = {'contrast_0': [0, 0, 1],
                  'contrast_1': [0, 1, 1],
                  }
    expected_output = {'contrast_0': [0, 0, 1],
                       'contrast_1': [0, 1, 1],
                       }
    actual_output = glmr._make_contrasts_dict(test_input)
    assert actual_output == expected_output


def test_make_contrasts_dict_with_list_of_lists():
    test_input = [[0, 0, 1], [0, 1, 0]]
    expected_output = {'[0, 0, 1]': [0, 0, 1],
                       '[0, 1, 0]': [0, 1, 0],
                       }
    actual_output = glmr._make_contrasts_dict(test_input)
    assert actual_output == expected_output


def test_make_contrasts_dict_with_list_of_arrays():
    test_input = [np.array([0, 0, 1]), np.array([0, 1, 0])]
    expected_output = {'[0 0 1]': np.array([0, 0, 1]),
                       '[0 1 0]': np.array([0, 1, 0]),
                       }
    actual_output = glmr._make_contrasts_dict(test_input)
    assert actual_output.keys() == expected_output.keys()
    for key in actual_output:
        assert np.array_equal(actual_output[key],
                              expected_output[key],
                              )



def test_make_contrasts_dict_with_list_of_ints():
    test_input = [1, 0, 1]
    expected_output = {'[1, 0, 1]': [1, 0, 1]}
    actual_output = glmr._make_contrasts_dict(test_input)
    assert np.array_equal(actual_output['[1, 0, 1]'],
                          expected_output['[1, 0, 1]'],
                          )

def test_make_contrasts_dict_with_array_of_ints():
    test_input = np.array([1, 0, 1])
    expected_output = {'[1 0 1]': np.array([1, 0, 1])}
    actual_output = glmr._make_contrasts_dict(test_input)
    assert expected_output.keys() == actual_output.keys()
    assert np.array_equal(actual_output['[1 0 1]'],
                          expected_output['[1 0 1]'],
                          )


def test_make_page_title_heading_with_contrasts_title_none():
    test_input = ({'contrast_0': [0, 0, 1],
                   'contrast_1': [0, 1, 1],
                   },
                  None,
                  )
    expected_output = ('Report: contrast_0, contrast_1',
                       'Statistical Report for contrasts',
                       'contrast_0, contrast_1',
                       )
    actual_output = glmr._make_headings(*test_input)
    assert actual_output == expected_output


def test_make_page_title_heading_with_contrasts_title_auto():
    test_input = ({'contrast_0': [0, 0, 1],
                   'contrast_1': [0, 1, 1],
                   },
                  'auto',
                  )
    expected_output = ('Report: contrast_0, contrast_1',
                       'Statistical Report for contrasts',
                       'contrast_0, contrast_1',
                       )
    actual_output = glmr._make_headings(*test_input)
    assert actual_output == expected_output


def test_make_page_title_heading_with_contrasts_title_custom():
    test_input = ({'contrast_0': [0, 0, 1],
                   'contrast_1': [0, 1, 1],
                   },
                  'Custom Title for report',
                  )
    expected_output = ('Custom Title for report',
                       'Custom Title for report',
                       '',
                       )
    actual_output = glmr._make_headings(*test_input)
    assert actual_output == expected_output


def test_make_page_title_heading_with_contrasts_none_title_custom():
    test_input = (None,
                  'Custom Title for report',
                  )
    expected_output = ('Custom Title for report',
                       'Custom Title for report',
                       '',
                       )
    actual_output = glmr._make_headings(*test_input)
    assert actual_output == expected_output


def _generate_img():
    mni_affine = np.array([[-2., 0., 0., 90.],
                           [0., 2., 0., -126.],
                           [0., 0., 2., -72.],
                           [0., 0., 0., 1.]])
    
    data_positive = np.zeros((7, 7, 3))
    rng = np.random.RandomState(42)
    data_rng = rng.rand(7, 7, 3)
    data_positive[1:-1, 2:-1, 1:] = data_rng[1:-1, 2:-1, 1:]
    
    return nib.Nifti1Image(data_positive, mni_affine)


def test_make_html_for_stat_maps():
    img = _generate_img()
    stat_map_html_code = glmr._make_stat_map_svg(stat_img=img,
                                                 threshold=4,
                                                 alpha=0.5,
                                                 cluster_threshold=0,
                                                 height_control='fdr',
                                                 bg_img=None,
                                                 display_mode='z',
                                                 plot_type='slice'
                                                 )
    assert True


def _make_dummy_contrasts_dmtx():
    frame_times = np.linspace(0, 127 * 1., 128)
    dmtx = make_first_level_design_matrix(frame_times,
                                          drift_model='polynomial',
                                          drift_order=3,
                                          )
    contrast = {'test': np.ones(4)}
    return contrast, dmtx


def test_make_contrast_matrix_html():
    contrast, dmtx = _make_dummy_contrasts_dmtx()
    contrast_plots = glmr._make_contrast_plots(contrast,
                                               [dmtx],
                                               )
    assert True
