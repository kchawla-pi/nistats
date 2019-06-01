import numpy as np

from nilearn import datasets
from nilearn.input_data import NiftiSpheresMasker

from nistats.first_level_model import FirstLevelModel
from nistats.design_matrix import make_first_level_design_matrix

from nistats.reporting.glm_reporter import generate_report


def make_zmaps(first_level_model, contrasts):
    """ Given a first model and contrasts, return the corresponding z-maps"""
    z_maps = {}
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        z_maps[contrast_id] = first_level_model.compute_contrast(
                contrast_val, output_type='z_score')
    return z_maps


def create_report_adhd_dmn():
    t_r = 2.
    slice_time_ref = 0.
    n_scans = 176
    pcc_coords = (0, -53, 26)
    adhd_dataset = datasets.fetch_adhd(n_subjects=1)
    seed_masker = NiftiSpheresMasker([pcc_coords], radius=10, detrend=True,
                                     standardize=True, low_pass=0.1,
                                     high_pass=0.01, t_r=2.,
                                     memory='nilearn_cache',
                                     memory_level=1, verbose=0)
    seed_time_series = seed_masker.fit_transform(adhd_dataset.func[0])
    frametimes = np.linspace(0, (n_scans - 1) * t_r, n_scans)
    design_matrix = make_first_level_design_matrix(frametimes, hrf_model='spm',
                                                   add_regs=seed_time_series,
                                                   add_reg_names=["pcc_seed"])
    dmn_contrast = np.array([1] + [0] * (design_matrix.shape[1] - 1))
    contrasts = {'seed_based_glm': dmn_contrast}
    
    first_level_model = FirstLevelModel(t_r=t_r, slice_time_ref=slice_time_ref)
    first_level_model = first_level_model.fit(run_imgs=adhd_dataset.func[0],
                                              design_matrices=design_matrix)
    
    output_filepath = 'generated_report.html'
    z_maps = make_zmaps(first_level_model, contrasts)
    generate_report(output_filepath, first_level_model,
                    contrasts=contrasts,
                    z_maps=z_maps,
                    mask=first_level_model.masker_.mask_img_,
                    design_matrices=first_level_model.design_matrices_,
                    bg_img=datasets.load_mni152_template(),
                    display_mode='z',
                    threshold=3.09,
                    scaled=True,
                    )
    # generate_subject_stats_report(output_filepath,
    #                               contrasts=contrasts,
    #                               z_maps=z_maps,
    #                               mask=first_level_model.masker_.mask_img_,
    #                               design_matrices=first_level_model.design_matrices_,
    #                               anat=datasets.load_mni152_template(),
    #                               )


if __name__ == '__main__':
    create_report_adhd_dmn()

