import numpy as np
from nose.tools import assert_true, assert_equal
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
# if __name__ == '__main__':
#     data_to_test_make_contrasts_dict = _make_data_to_test_make_contrasts_dict()
#     test_make_contrasts_dict(data_to_test_make_contrasts_dict)
