import numpy as np

from nilearn import datasets
from nilearn.input_data import NiftiSpheresMasker

from nistats.first_level_model import FirstLevelModel
from nistats.design_matrix import make_first_level_design_matrix

from nistats.reporting.glm_reporter import make_glm_report


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
    
    output_filepath = 'generated_report_flm_adhd_dmn.html'
    report = make_glm_report(
            first_level_model,
            contrasts=contrasts,
            title='ADHD DMN Report',
            min_distance=8.,
            cluster_threshold=15,
            plot_type='glass',
            )
    # report.open_in_browser()
    report.save_as_html(output_filepath)


if __name__ == '__main__':
    create_report_adhd_dmn()
