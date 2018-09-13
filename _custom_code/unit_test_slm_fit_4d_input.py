import nilearn
import os
import pandas
from nistats.second_level_model import SecondLevelModel, validate_input_as_4d_niimg
from nistats.first_level_model import FirstLevelModel


def test_4d_niimg():
    img1_4d = nilearn.image.load_img(os.path.expanduser(
            '~/nilearn_data/adhd/data/0010128/0010128_rest_tshift_RPI_voreg_mni.nii.gz'))
    c = pandas.DataFrame([[1]] * img1_4d.shape[3], columns=['intercept'])
    slm = SecondLevelModel(smoothing_fwhm=2.0)
    try:
        slm.fit(second_level_input=img1_4d, design_matrix=c)
    except Exception as err:
        print('4D niimg:', err)


def test_4d_niimg_as_list():
    img2_4d = nilearn.image.load_img(os.path.expanduser(
            '~/nilearn_data/adhd/data/2497695/2497695_rest_tshift_RPI_voreg_mni.nii.gz'))
    
    c = pandas.DataFrame([[1]] * img2_4d.shape[3], columns=['intercept'])
    slm = SecondLevelModel(smoothing_fwhm=2.0)
    try:
        slm.fit(second_level_input=[img2_4d], design_matrix=c)
    except Exception as err:
        print('[4D niimg]:', err)


def test_3d_niimgs_list():
    img3_3d = nilearn.image.load_img(os.path.expanduser(
            '~/nilearn_data/destrieux_2009/destrieux2009_rois_lateralized.nii.gz'))
    multi_3d = [img3_3d, img3_3d]  # List of > 1 3D niimgs.
    c = pandas.DataFrame([[1]] * len(multi_3d), columns=['intercept'])
    slm = SecondLevelModel(smoothing_fwhm=2.0)
    try:
        slm.fit(second_level_input=multi_3d, design_matrix=c)
    except Exception as err:
        print('[3d niimgs]:', err)


# path pattern to dir with multiple 3D niimgs.
def test_3d_niimgs_dir_path_pattern():
    imgs_3d_path = os.path.expanduser('~/nilearn_data/destrieux_2009/*.nii.gz')
    slm = SecondLevelModel(smoothing_fwhm=2.0)
    try:
        slm.fit(second_level_input=imgs_3d_path)
    except Exception as err:
        print('[3d niimgs] path:', err)


def test_path_4d_niimg():
    img_4d_path = os.path.expanduser('~/nilearn_data/adhd/data/0010128/0010128_rest_tshift_RPI_voreg_mni.nii.gz'),  # path to 4D niimg.
    slm = SecondLevelModel(smoothing_fwhm=2.0)
    try:
        slm.fit(second_level_input=img_4d_path)
    except Exception as err:
        print('4d niimg path:', err)


def test_4d_niimgs_list():
    img2_4d = nilearn.image.load_img(os.path.expanduser(
            '~/nilearn_data/adhd/data/2497695/2497695_rest_tshift_RPI_voreg_mni.nii.gz'))
    multi_4d_niimgs = (img2_4d, img2_4d)
    c = pandas.DataFrame([[1]] * len(multi_4d_niimgs), columns=['intercept'])
    slm = SecondLevelModel(smoothing_fwhm=2.0)
    try:
        slm.fit(second_level_input=multi_4d_niimgs, design_matrix=c)
    except Exception as err:
        print('[4D niimgs]:', err)


def test_3d_niimg():
    img3_3d = nilearn.image.load_img(os.path.expanduser(
            '~/nilearn_data/destrieux_2009/destrieux2009_rois_lateralized.nii.gz'))
    c = pandas.DataFrame([[1]] * 1, columns=['intercept'])
    slm = SecondLevelModel(smoothing_fwhm=2.0)
    try:
        slm.fit(second_level_input=img3_3d, design_matrix=c)
    except Exception as err:
        print('3d niimg:', err)


def test_6d():
    img1_4d = nilearn.image.load_img(os.path.expanduser(
            '~/nilearn_data/adhd/data/0010128/0010128_rest_tshift_RPI_voreg_mni.nii.gz'))
    slm = SecondLevelModel(smoothing_fwhm=2.0)
    try:
        slm.fit(([img1_4d],))
    except Exception as err:
        print('6D:', err)


def test_flm_object():
    img1_4d = nilearn.image.load_img(os.path.expanduser(
            '~/nilearn_data/adhd/data/0010128/0010128_rest_tshift_RPI_voreg_mni.nii.gz'))
    c = pandas.DataFrame([[1]] * img1_4d.shape[3], columns=['intercept'])
    flm = FirstLevelModel(subject_label='01').fit(img1_4d, design_matrices=c)
    
    slm = SecondLevelModel(smoothing_fwhm=2.0)
    try:
        slm.fit(second_level_input=flm, design_matrix=c)
    except Exception as err:
        print('FLM with design matrix:', err)


def test_flms_list():
    img1_4d = nilearn.image.load_img(os.path.expanduser(
            '~/nilearn_data/adhd/data/0010128/0010128_rest_tshift_RPI_voreg_mni.nii.gz'))
    c = pandas.DataFrame([[1]] * img1_4d.shape[3], columns=['intercept'])
    flm = FirstLevelModel(subject_label='01').fit(img1_4d, design_matrices=c)
    multi_flms = [flm, flm, flm]
    slm = SecondLevelModel(smoothing_fwhm=2.0)
    try:
        slm.fit(second_level_input=multi_flms, design_matrix=c)
    except Exception as err:
        print('[FLMs] with design matrix:', err)
    else:
        print('[FLMs]: No error')


def test_string():
    slm = SecondLevelModel(smoothing_fwhm=2.0)
    try:
        slm.fit('string')
    except Exception as err:
        print(err)


if __name__ == '__main__':
    print('-'*12, '\n', 'Expecting slice error')
    test_4d_niimg()
    test_4d_niimg_as_list()
    test_3d_niimgs_list()
    test_3d_niimgs_dir_path_pattern()
    print('-' * 12, '\n', 'Expecting incorrect dimension error')
    test_4d_niimgs_list()
    test_3d_niimg()
    test_6d()
    print('-' * 12, '\n', 'Expecting some other error')
    test_flm_object()
    test_string()
    print('-' * 12, '\n', 'Expecting no error or maybe some other error???')
    test_flms_list()
    quit()


# pd_dataframe_test = validate_input_as_4d_niimg(c)
#


# pd_dataframe_test,  # Pandas dataframe.

