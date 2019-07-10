import numpy as np
from nose.tools import assert_true, assert_equal

from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import FirstLevelModel
from nistats.reporting import glm_reporter as glmr


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
        
        
def test_make_contrast_matrix_html():
    frame_times = np.linspace(0, 127 * 1., 128)
    dmtx = make_first_level_design_matrix(frame_times,
                                          drift_model='polynomial',
                                          drift_order=3,
                                          )
    contrast = {'test': np.ones(4)}
    with open('data_for_testing_glm_reporter/expected_contrast_plot.txt') as f:
        expected_contrast_plot_text = f.read()
    flm = FirstLevelModel()
    flm.design_matrices_ = [dmtx]
    contrast_plots = glmr._make_dict_of_contrast_plots(contrast, flm)
    contrast_plots_text = ['{}<p>{}'.format(key, item)
                           for key, item in contrast_plots.items()
                           ]
    contrast_plots_html = '<p>'.join(contrast_plots_text)

    assert_equal(expected_contrast_plot_text, contrast_plots_html)


# if __name__ == '__main__':
#     test_make_contrast_matrix_html()
#     data_to_test_make_contrasts_dict = _make_data_to_test_make_contrasts_dict()
#     test_make_contrasts_dict(data_to_test_make_contrasts_dict)
