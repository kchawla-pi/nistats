import nilearn
import os
import pandas
from nistats.second_level_model import SecondLevelModel, validate_input_as_4d_niimg
from nistats.first_level_model import FirstLevelModel


# 4D niimg object.
img1_4d = nilearn.image.load_img(os.path.expanduser(
        '~/nilearn_data/adhd/data/0010128/0010128_rest_tshift_RPI_voreg_mni.nii.gz'))
c = pandas.DataFrame([[1]] * img1_4d.shape[3], columns=['intercept'])
flm = FirstLevelModel(subject_label='01').fit(img1_4d, design_matrices=c)


# 4D niimg object.
img2_4d = nilearn.image.load_img(os.path.expanduser(
        '~/nilearn_data/adhd/data/2497695/2497695_rest_tshift_RPI_voreg_mni.nii.gz'))
c = pandas.DataFrame([[1]] * img2_4d.shape[3], columns=['intercept'])
flm = FirstLevelModel(subject_label='01').fit(img2_4d, design_matrices=c)


# 3D niimg object.
img3_3d = nilearn.image.load_img(os.path.expanduser(
        '~/nilearn_data/destrieux_2009/destrieux2009_rois_lateralized.nii.gz'))
c = pandas.DataFrame([[1]] * img3_3d.shape[3], columns=['intercept'])
flm = FirstLevelModel(subject_label='01').fit(img3_3d, design_matrices=c)


test_case = [img3_3d, img3_3d],  # List of > 1 3D niimgs.
c = pandas.DataFrame([[1]] * len(test_case), columns=['intercept'])
flm = FirstLevelModel(subject_label='01').fit(img3_3d, design_matrices=c)


test_case = os.path.expanduser('~/nilearn_data/adhd/data/0010128/0010128_rest_tshift_RPI_voreg_mni.nii.gz'),  # path to 4D niimg.

# pd_dataframe_test = validate_input_as_4d_niimg(c)
#

test_cases = (
    
    os.path.expanduser('~/nilearn_data/destrieux_2009/*.nii.gz'),  # path pattern to dir with multiple 3D niimgs.
    # flm,  # FirstLevelModel object.
    [img1_4d],  # list with only 1 element, a 4D niimg.
    [img1_4d, img2_4d],  # list with 2 4D niimgs.
    img3_3d,  # 3D niimg.
    '1',  # string.
    # pd_dataframe_test,  # Pandas dataframe.
    )

