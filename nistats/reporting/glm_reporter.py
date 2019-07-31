import io
import string
import os

from collections import OrderedDict
try:
    from urllib.parse import quote
except ImportError:
    from urllib import quote

import pandas as pd

from matplotlib import pyplot as plt
from nilearn.plotting import (plot_glass_brain,
                              plot_roi,
                              plot_stat_map,
                              )
from nilearn.plotting.img_plotting import MNI152TEMPLATE
from nilearn.plotting.js_plotting_utils import HTMLDocument

from nistats.reporting import (plot_contrast_matrix,
                               plot_design_matrix,
                               get_clusters_table,
                               )
from nistats.thresholding import map_threshold


html_template_root_path = os.path.dirname(__file__)


def make_glm_report(
        model,
        contrasts,
        title='auto',
        roi_img=None,
        bg_img=MNI152TEMPLATE,
        threshold=3.09,
        alpha=0.01,
        cluster_threshold=0,
        height_control='fpr',
        min_distance=8.,
        plot_type='stat',
        display_mode=None,
        cut_coords=None,
        nb_width=1600,
        nb_height=800,
        ):
    """ Returns HTMLDocument object for a report which shows
    all important aspects of a fitted GLM.
    The object can be opened in a browser or saved to disk.
    
    Usage:
        report = make_glm_report(model, contrasts)
        report.open_in_browser()
        report.save_as_html(destination_path)
    
    Parameters
    ----------
    model: FirstLevelModel or SecondLevelModel object
        A fitted first or second level model object.
        
    contrasts: Dict[string, ndarray] , String
    
    title: str, default 'auto'
        Text to represnt the web page's title and primary heading.
        
    threshold: float
        Cluster forming threshold in same scale as `stat_img` (either a
        p-value or z-scale value).
        Default is 3.09
        
    alpha: float
        P- value for clustering.

    cluster_threshold : int, optional
        Cluster size threshold, in voxels.
        Default is 0
        
    height_control: str

    min_distance: float, optional
        Minimum distance between subpeaks in mm. Default is 8 mm.
        
    bg_img: Nifti image
        Default is the MNI152 template
        
    display_mode: String, optional
        Default is 'z' if plot_type is 'stat'; 'ortho' if plot_type is 'glass'.
        
    plot_type: String. ['stat' (default)| 'glass']
        Specifies the type of plot to be drawn for the statistical maps.
        
    Returns
    -------
    report_text: HTMLDocument Object
        Contains the HTML code for the GLM Report.
    """
    if not display_mode:
        if plot_type == 'stat':
            display_mode = 'z'
        elif plot_type == 'glass':
            display_mode = 'lzry'
            
    try:
        design_matrices = model.design_matrices_
    except AttributeError:
        design_matrices = [model.design_matrix_]

    html_template_path = os.path.join(html_template_root_path,
                                      'report_template.html')
    with open(html_template_path) as html_file_obj:
        html_template_text = html_file_obj.read()
    report_template = string.Template(html_template_text)
    
    contrasts = _make_contrasts_dict(contrasts)
    contrast_plots = _make_dict_of_contrast_plots(contrasts, design_matrices)
    page_title, page_heading_1, page_heading_2 = _make_page_title_heading(
        contrasts,
        title,
        )
    with pd.option_context('display.max_colwidth', 100):
        model_attributes_html = _make_model_attributes_html_table(model)
    statistical_maps = make_statistical_maps(model, contrasts)
    html_design_matrices = _make_html_for_design_matrices(design_matrices)
    roi_plot_html_code = _make_roi_plot(roi_img, bg_img)
    all_components = _make_report_components(
            statistical_maps=statistical_maps,
            contrasts_plots=contrast_plots,
            threshold=threshold,
            alpha=alpha,
            cluster_threshold=cluster_threshold,
            height_control=height_control,
            min_distance=min_distance,
            bg_img=bg_img,
            display_mode=display_mode,
            plot_type=plot_type,
            )
    all_components_text = '\n'.join(all_components)
    report_values = {'page_title': page_title,
                     'page_heading_1': page_heading_1,
                     'page_heading_2': page_heading_2,
                     'model_attributes': model_attributes_html,
                     'all_contrasts_with_plots': ''.join(contrast_plots.values()),
                     'design_matrices': html_design_matrices,
                     'roi_plot': roi_plot_html_code,
                     'component': all_components_text,
                     }
    report_text = report_template.safe_substitute(**report_values)
    report = HTMLDocument(report_text)
    report.width = nb_width  # better visual experience in Jupyter Notebooks.
    report.height = nb_height
    return report


def _make_contrasts_dict(contrasts):
    """ Accepts contrasts and returns a dict of them.
    with the names of contrasts as keys.
    
    If contrasts is:
      dict then returns it unchanged.
      
      string or list/tuple of strings, returns a dict where key==values
    
    Parameters
    ----------
    contrasts: str, List/Tuple[str], Dict[str, str or np.array]
        Contrast information
    
    Returns
    -------
    contrasts: Dict[str, np.array or str]
        Contrast information, as a dict
    """
    contrasts = [contrasts] if isinstance(contrasts, str) else contrasts
    if not isinstance(contrasts, dict):
        contrasts = {contrast_: contrast_ for contrast_ in contrasts}
    return contrasts


def _make_dict_of_contrast_plots(contrasts, design_matrices):
    """
    Accepts dict of contrasts and First or Second Level Model
    with fitted design matrices and generates
    a dict of contrast name & svg code for corresponding contrast plot.
    
    Parameters
    ----------
    contrasts: Dict[str, np.array or str]
        Contrast information, as a dict
    
    design_matrices: design_matrices

    Returns
    -------
    contrast_plots: Dict[str, svg img]
        Dict of contrast name and svg code for corresponding contrast plot.
    """
    all_contrasts_plots = {}
    contrast_template_path = os.path.join(html_template_root_path,
                                            'contrast_template.html'
                                            )
    with open(contrast_template_path) as html_template_obj:
        contrast_template_text = html_template_obj.read()

    for design_matrix in design_matrices:
        for contrast_name, contrast_data in contrasts.items():
            contrast_text_ = string.Template(contrast_template_text)
            contrast_plot = plot_contrast_matrix(contrast_data, design_matrix)
            contrast_plot.set_xlabel(contrast_name)
            contrast_plot.figure.set_tight_layout(True)
            contrast_plot.figure.set_figheight(2)
            url_contrast_plot_svg = make_svg_image_data_url(contrast_plot)
            contrasts_for_subsitution = {'contrast_plot': url_contrast_plot_svg}
            contrast_text_ = contrast_text_.safe_substitute(
                    contrasts_for_subsitution
                    )
            all_contrasts_plots[contrast_name] = contrast_text_
    return all_contrasts_plots
    
    
def _make_page_title_heading(contrasts, title):
    """ Creates report page title, heading & sub-heading
     using title text or contrast names.
    Accepts contrasts and user supplied title string.
    
    If title is not in (None, 'auto'), page title == heading, no sub-heading
    
    Parameters
    ----------
    contrasts: Dict[str, np.array or str] or List/Tuple[str] or String
        Needed for contrast names.
    
    title: str
        User supplied text for HTML Page title and primary heading.
    
    Returns
    -------
    (HTML page title, heading, sub-heading): Tuple[str, str, str]
    """
    if title not in ('auto', None):
        return title, title, ''
    else:
        if isinstance(contrasts, str):
            contrasts_text = contrasts
        else:
            contrasts_names = sorted(contrasts)
            contrasts_text = ', '.join(contrasts_names)
        page_title = 'Report: {}'.format(contrasts_text)
        page_heading_1 = 'Statistical Report for contrasts'
        page_heading_2 = '{}'.format(contrasts_text)
        return page_title, page_heading_1, page_heading_2


def _make_model_attributes_html_table(model):
    """ Returns an HTML table with pertinent model attributes & information.
    
    Parameters
    ----------
    model: FirstLevelModel or SecondLevelModel object.
    
    Returns
    -------
    HTML Table: String
        HTML code for creating a table.
    """
    selected_attributes = [
        'subject_label',
        'mask_img',
        'drift_model',
        'hrf_model',
        'standardize',
        'noise_model',
        'min_onset',
        't_r',
        'labels_',
        'high_pass',
        'target_shape',
        'signal_scaling',
        'drift_order',
        'scaling_axis',
        'smoothing_fwhm',
        'target_affine',
        'slice_time_ref',
        'fir_delays',
        ]
    selected_model_attributes_values = {
        attr_name: model.__dict__[attr_name]
        for attr_name in selected_attributes
        if attr_name in model.__dict__
        }
    model_attributes_table = pd.DataFrame.from_dict(selected_model_attributes_values,
                                                    orient='index',
                                                    )
    model_attributes_table = model_attributes_table.to_html(header=False,
                                                            sparsify=False,
                                                            )
    return model_attributes_table


def make_statistical_maps(model, contrasts):
    """ Given a model and contrasts, return the corresponding z-maps
    
    Parameters
    ----------
    model: FirstLevelModel or SecondLevelModel object
    
    contrasts: Dict[str, ndarray or str]
        Dict of contrasts
    
    Returns
    -------
    statistical_maps: Dict[str, niimg]
        Dict of statistical maps keyed to contrast names.
    """
    statistical_maps = {contrast_id: model.compute_contrast(contrast_val)
                        for contrast_id, contrast_val in contrasts.items()
                        }
    return statistical_maps


def _make_html_for_design_matrices(design_matrices):
    """ Accepts a FirstLevelModel or SecondLevelModel object
    with fitted design matrices & generates HTML code
    to insert their plots into the report.
    
    Parameters
    ----------
    desin_matrices: design_matrices
        
    Returns
    -------
    html_design_matrices: String
        HTML code for the plotted design matrices,
        to be inserted into the HTML template.
    """
    html_design_matrices = []
    dmtx_template_path = os.path.join(html_template_root_path,
                                            'design_matrix_template.html'
                                            )
    with open(dmtx_template_path) as html_template_obj:
        dmtx_template_text = html_template_obj.read()

    for dmtx_count, design_matrix in enumerate(design_matrices, start=1):
        dmtx_text_ = string.Template(dmtx_template_text)
        dmtx_plot = plot_design_matrix(design_matrix)
        plt.title(dmtx_count, y=0.987)
        url_design_matrix_svg = make_svg_image_data_url(dmtx_plot)
        dmtx_text_ = dmtx_text_.safe_substitute(
                {'design_matrix': url_design_matrix_svg}
                )
        html_design_matrices.append(dmtx_text_)
    html_design_matrices = ''.join(html_design_matrices)
    return html_design_matrices


def make_svg_image_data_url(plot):
    with io.StringIO() as buffer:
        try:
            plot.figure.savefig(buffer, format='svg')
        except AttributeError:
            plot.savefig(buffer, format='svg')
        plot_svg = buffer.getvalue()
    url_plot_svg = quote(plot_svg.decode('utf8'))
    return url_plot_svg
    

def _make_roi_plot(roi_img, bg_img):
    """
    Accepts an ROI image and background image
    to create svg code for a ROI plot.
    
    Parameters
    ----------
    roi_img: niimg
        ROI mask image
        
    bg_img: niimg
        Background image

    Returns
    -------
    roi_plot_html_code: str
        SVG code for the ROI plot, can be inlined into an HTML document
    """
    if roi_img:
        roi_plot = plot_roi(roi_img=roi_img, bg_img=bg_img)
        roi_plot_html_code = make_svg_image_data_url(plt.gcf())
    else:
        roi_plot_html_code = 'Pass the mask with the `roi_img` parameter to plot the ROI'
    return roi_plot_html_code
    

def _make_report_components(statistical_maps, contrasts_plots, threshold, alpha,
                            cluster_threshold, height_control, min_distance, bg_img,
                            display_mode, plot_type):
    """ Populates a smaller HTML sub-template with the proper values,
     make a list containing one or more of such components
     & returns the list to be inserted into the HTML Report Template.
    Each component contains the HTML code for
    a contrast & its corresponding statistical maps & cluster table;
    
    Parameters
    ----------
    statistical_maps: Nifti images
    
    contrasts_plots: Dict[str, ndarray or str]
    
    threshold: float
    
    alpha: float
    
    bg_img: Nifti image
    
    display_mode: string
    
    Returns
    -------
    all_components: String
        HTML code representing each set of
        contrast, statistical map, cluster table.
    """
    all_components = []
    components_template_path = os.path.join(html_template_root_path,
                                            'report_components_template.html'
                                            )
    with open(components_template_path) as html_template_obj:
        components_template_text = html_template_obj.read()
    for contrast_name, stat_map_img in statistical_maps.items():
        component_text_ = string.Template(components_template_text)
        stat_map_html_code = _make_html_for_stat_maps(
                statistical_map_img=stat_map_img,
                threshold=threshold,
                alpha=alpha,
                cluster_threshold=cluster_threshold,
                height_control=height_control,
                bg_img=bg_img,
                display_mode=display_mode,
                plot_type=plot_type,
                )
        cluster_table_details_html, cluster_table_html = (
            _make_html_for_cluster_table(statistical_map_img=stat_map_img,
                                         threshold=threshold,
                                         alpha=alpha,
                                         cluster_threshold=cluster_threshold,
                                         height_control=height_control,
                                         min_distance=min_distance,
                                         )
        )
        components_values = {
            'contrast_name': contrast_name,
            'contrast_plot': contrasts_plots[contrast_name],
            'stat_map_img': stat_map_html_code,
            'cluster_table_details': cluster_table_details_html,
            'cluster_table': cluster_table_html,
            }
        component_text_ = component_text_.safe_substitute(**components_values)
        all_components.append(component_text_)
    return all_components


def _make_html_for_stat_maps(statistical_map_img,
                             threshold,
                             alpha,
                             cluster_threshold,
                             height_control,
                             bg_img,
                             display_mode,
                             plot_type,
                             ):
    """ Generates string of HTML code for a statistical map.
    
    Parameters
    ----------
    statistical_map_img: Nifti image
    
    threshold: float
    
    alpha: float
    
    bg_img: Nifti image
    
    display_mode: String
    
    Returns
    -------
    stat_map_html_code: String
        String of HTML code representing a statistical map.
    """
    thresholded_stat_map_img, _ = map_threshold(statistical_map_img,
                                                threshold=threshold,
                                                alpha=alpha,
                                                cluster_threshold=cluster_threshold,
                                                height_control=height_control,
                                                )
    if plot_type == 'glass':
        stat_map_plot = plot_glass_brain(thresholded_stat_map_img,
                                         display_mode=display_mode,
                                         colorbar=True,
                                         plot_abs=False,
                                         )
    else:
        stat_map_plot = plot_stat_map(thresholded_stat_map_img,
                                      bg_img=bg_img,
                                      display_mode=display_mode,
                                      )
    buffer = io.StringIO()
    plt.savefig(buffer, format='svg')
    stat_map_html_code = buffer.getvalue()
    return stat_map_html_code


def _make_html_for_cluster_table(statistical_map_img,
                                 threshold,
                                 alpha,
                                 cluster_threshold,
                                 height_control,
                                 min_distance
                                 ):
    """ Generates string of HTML code for a cluster table.

    Parameters
    ----------
    statistical_map_img: Nifti image
    
    thrshold: float
    
    alpha: float
    
    cluster_threshold: int
    
    height_control: str
    
    min_distance: float

    Returns
    -------
    single_cluster_table_html_code: String
        HTML code representing a cluster table.
    """
    cluster_table = get_clusters_table(statistical_map_img,
                                       stat_threshold=threshold,
                                       cluster_threshold=cluster_threshold,
                                       min_distance=min_distance,
                                       )
    cluster_table_details = OrderedDict()
    cluster_table_details.update({'Threshold Z': threshold})
    cluster_table_details.update({'Cluster size threshold (voxels)':
                                      cluster_threshold}
                                 )
    cluster_table_details.update({'Minimum distance (mm)': min_distance})
    cluster_table_details.update({'Height control': height_control})
    cluster_table_details.update({'Cluster Level p-value Threshold': alpha})
    try:
        cluster_table_details_html = pd.DataFrame.from_dict(
                cluster_table_details, orient='index').to_html(border=0,
                                                               header=False,
                                                               )
    except TypeError:  # pandas < 0.19
        cluster_table_details_html = pd.DataFrame.from_dict(
                cluster_table_details, orient='index').to_html(
                                                               header=False,
                                                               )
    with pd.option_context('display.precision', 2):
        single_cluster_table_html_code = cluster_table.to_html()
    return cluster_table_details_html, single_cluster_table_html_code
