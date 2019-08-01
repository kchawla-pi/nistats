import os

import nibabel as nib
import numpy as np

from nose import SkipTest

import pandas as pd
from nose.tools import assert_equal

from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import FirstLevelModel
from nistats.reporting import glm_reporter as glmr
from nistats import datasets
from numpy.testing import dec

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
    report = glmr.make_glm_report(fmri_glm, contrasts, bg_img=mean_img_,
                                  roi_img=data['mask'])
    benchmark_filepath = 'fiac_flm_report_benchmark.html'
    
    
def _make_data_to_test_make_contrasts_dict():
    test_cases = [{'test_input': 'StopSuccess - Go',
                   'expected_output': {'StopSuccess - Go': 'StopSuccess - Go'}
                   },
                  {'test_input': ['contrast_name_0', 'contrast_name_1'],
                   'expected_output': {'contrast_name_0': 'contrast_name_0',
                                       'contrast_name_1': 'contrast_name_1',
                                       }
                   },
                  {'test_input': {'contrast_0': [0, 0, 1],
                                  'contrast_1': [0, 1, 1],
                                  },
                   'expected_output': {'contrast_0': [0, 0, 1],
                                       'contrast_1': [0, 1, 1],
                                       },
                   },
                  ]
    return test_cases


def test_make_contrasts_dict(test_cases=_make_data_to_test_make_contrasts_dict()):
    for test_case_ in test_cases:
        actual_output = glmr._make_contrasts_dict(test_case_['test_input'])
        assert_equal(test_case_['expected_output'], actual_output)


def _make_data_to_test_make_page_title_heading():
    test_cases = [
        {'test_input': ({'contrast_0': [0, 0, 1], 'contrast_1': [0, 1, 1]},
                        None),
         'expected_output': ('Report: contrast_0, contrast_1',
                             'Statistical Report for contrasts',
                             'contrast_0, contrast_1',
                             )
         },
        {'test_input': ({'contrast_0': [0, 0, 1], 'contrast_1': [0, 1, 1]},
                        'auto'),
         'expected_output': ('Report: contrast_0, contrast_1',
                             'Statistical Report for contrasts',
                             'contrast_0, contrast_1',
                             )
         },
        {'test_input': ({'contrast_0': [0, 0, 1], 'contrast_1': [0, 1, 1]},
                        'Custom Title for report'),
         'expected_output': ('Custom Title for report',
                             'Custom Title for report',
                             '',
                             )
         },
        {'test_input': (None,
                        'Custom Title for report'),
         'expected_output': ('Custom Title for report',
                             'Custom Title for report',
                             '',
                             )
         },
    
        ]
    return test_cases
    
    
def test_make_page_title_heading(test_cases=_make_data_to_test_make_page_title_heading()):
    for test_case_ in test_cases:
        actual_output = glmr._make_page_title_heading(*test_case_['test_input'])
        assert_equal(test_case_['expected_output'], actual_output)


def test_make_html_for_cluster_table():
    shape = (9, 10, 11)
    data = np.zeros(shape)
    data[2:4, 5:7, 6:8] = 5.
    stat_img = nib.Nifti1Image(data, np.eye(4))
    table_details_html, table_html_code = glmr._make_html_for_cluster_table(
            stat_img, 4, 0.5, 0, 'fdr', 8,
            )
    
    expected_html_fragments_table_details = [
        '<th>Threshold Z</th>',
        '<th>Cluster size threshold (voxels)</th>',
        '<th>Minimum distance (mm)</th>',
        '<th>Height control</th>',
        '<th>Cluster Level p-value Threshold</th>',
        ]
    table_details_check = [
        fragment in table_details_html
        for fragment in expected_html_fragments_table_details
        ]
    assert all(table_details_check)
    
    expected_html_fragments_table_code = [
        '<th>Cluster ID</th>',
        '<th>X</th>',
        '<th>Y</th>',
        '<th>Z</th>',
        '<th>Peak Stat</th>',
        '<th>Cluster Size (mm3)</th>',
        '<th>0</th>',
        '<td>1</td>',
        '<td>2.0</td>',
        '<td>6.0</td>',
        '<td>6.0</td>',
        '<td>5.0</td>',
        '<td>8</td>',
        ]
    table_code_check = [
        fragment in table_details_html
        for fragment in expected_html_fragments_table_details
        ]
    assert all(table_code_check)


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
    stat_map_html_code = glmr._make_html_for_stat_maps(stat_img=img,
                                                       threshold=4,
                                                       alpha=0.5,
                                                       cluster_threshold=0,
                                                       height_control='fdr',
                                                       bg_img=None,
                                                       display_mode='z',
                                                       plot_type=None,
                                                       )
    assert True


def _make_dummy_contrasts_dmtx_model():
    frame_times = np.linspace(0, 127 * 1., 128)
    dmtx = make_first_level_design_matrix(frame_times,
                                          drift_model='polynomial',
                                          drift_order=3,
                                          )
    contrast = {'test': np.ones(4)}
    flm = FirstLevelModel()
    flm.design_matrices_ = [dmtx]
    return contrast, dmtx, flm


def test_make_contrast_matrix_html():
    contrast, dmtx, flm = _make_dummy_contrasts_dmtx_model()
    contrast_plots = glmr._make_dict_of_contrast_plots(contrast,
                                                       flm.design_matrices_,
                                                       )
    assert True


# if __name__ == '__main__':
#     test_make_contrast_matrix_html()
    # data_to_test_make_contrasts_dict = _make_data_to_test_make_contrasts_dict()
    # test_make_contrasts_dict(data_to_test_make_contrasts_dict)
