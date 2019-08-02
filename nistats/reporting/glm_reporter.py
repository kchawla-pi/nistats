import io
import string
import os

from collections import OrderedDict

import numpy as np

try:
    from urllib.parse import quote
except ImportError:
    from urllib import quote

try:
    from html import escape
except ImportError:
    from cgi import escape

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

html_template_root_path = os.path.join(os.path.dirname(__file__),
                                       'glm_reporter_templates')


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
        plot_type='slice',
        display_mode=None,
        nb_width=1600,
        nb_height=800,
        ):
    """ Returns HTMLDocument object for a report which shows
    all important aspects of a fitted GLM.
    The object can be opened in a browser, displayed in a notebook,
    or saved to disk as a portable HTML file.
    
    Usage:
        report = make_glm_report(model, contrasts)
        report.open_in_browser()
        report.save_as_html(destination_path)
    
    Parameters
    ----------
    model: FirstLevelModel or SecondLevelModel object
        A fitted first or second level model object.
        Must have the computed design matrix(ces).
        
    contrasts: Dict[string, ndarray] , String, List[ndarrays]
        Contrasts information for a first or second level model.
        Corresponds to the contrast_def for the FirstLevelModel [1]_
        & second_level_contrast for a SecondLevelModel [2]_ .
    
    title: String, default 'auto'
        Text to represent the web page's title and primary heading.
        If 'auto', uses the contrast titles to generate a title.
        
    roi_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The ROI/mask image, it could be binary mask or an atlas or ROIs
        with integer values.
    
    bg_img : Niimg-like object
        Default is the MNI152 template
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image that the ROI/mask will be plotted on top of.
        To turn off background image, just pass "bg_img=None".
        
    threshold: float
        Default is 3.09
        Cluster forming threshold in same scale as `stat_img` (either a
        p-value or z-scale value).
        
    alpha: float
        Default is 0.01
        Number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.

    cluster_threshold : int, optional
        Default is 0
        Cluster size threshold, in voxels.
        
    height_control: string
        false positive control meaning of cluster forming
        threshold: 'fpr' (default)|'fdr'|'bonferroni'|None
    
    min_distance: `float`
        For display purposes only.
        Minimum distance between subpeaks in mm. Default is 8 mm.
        
    plot_type: String. ['slice' (default)| 'glass']
        Specifies the type of plot to be drawn for the statistical maps.
        
    display_mode: string
        Default is 'z' if plot_type is 'slice'; '
        ortho' if plot_type is 'glass'.
        
        Choose the direction of the cuts:
        'x' - sagittal, 'y' - coronal, 'z' - axial,
        'l' - sagittal left hemisphere only,
        'r' - sagittal right hemisphere only,
        'ortho' - three cuts are performed in orthogonal directions.
        
        Possible values are:
        'ortho', 'x', 'y', 'z', 'xz', 'yx', 'yz',
        'l', 'r', 'lr', 'lzr', 'lyr', 'lzry', 'lyrz'.
        
    nb_width: int
        Default is 1600 (px).
        Specifies width (in pixels) of report window within the notebook.
        Only applicable when inserting the report into a Jupyter notebook.
        
    nb_height: int
        Default is 800 (px).
        Specifies height (in pixels) of report window within the notebook.
        Only applicable when inserting the report into a Jupyter notebook.
    
    Returns
    -------
    report_text: HTMLDocument Object
        Contains the HTML code for the GLM Report [3]_ .
        
    See Also
    --------
    .. [1] nistats.first_level_model.FirstLevelModel.compute_contrast
    .. [2] nistats.second_level_model.SecondLevelModel.compute_contrast
    .. [3] nilearn.plotting.js_plotting_utils.HTMLDocument

    """
    if not display_mode:
        if plot_type == 'slice':
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
    contrast_plots = _make_contrast_plots(contrasts, design_matrices)
    page_title, page_heading_1, page_heading_2 = _make_headings(
            contrasts,
            title,
            )
    with pd.option_context('display.max_colwidth', 100):
        model_attributes_html = _make_attributes_table(model)
    statistical_maps = make_stat_maps(model, contrasts)
    html_design_matrices = _make_svg_url_design_matrices(design_matrices)
    roi_plot_html_code = _make_roi_svg(roi_img, bg_img)
    all_components = _make_report_components(
            stat_img=statistical_maps,
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
    report_values = {'page_title': escape(page_title),
                     'page_heading_1': page_heading_1,
                     'page_heading_2': page_heading_2,
                     'model_attributes': model_attributes_html,
                     'all_contrasts_with_plots': ''.join(
                         contrast_plots.values()),
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
      
      string or collection of Strings or Sequence[int],
      returns a dict where key==values
    
    Parameters
    ----------
    contrasts: String or Collection[str or Sequence[Int]] or Dict[str, str or np.array]
        Contrast information. Can be a dict of the form
         {'contrast_name_1': contrast list/array1, 'contrast_name_1': contrast list/array1}
         ['contast_name_1', 'contast_name_2', ...]
         [contrast_array_1, contrast_array_2, ...]
         'contrast_name'
    
    Returns
    -------
    contrasts: Dict[str, np.array or str]
        Contrast information, as a dict in the form
            {'contrast_title_1': contrast_info_1/title_1, ...}
    """
    contrasts = [contrasts] if isinstance(contrasts, str) else contrasts
    if not isinstance(contrasts, dict):
        contrasts = {str(contrast_): contrast_ for contrast_ in contrasts}
    return contrasts


def _make_contrast_plots(contrasts, design_matrices):
    """
    Accepts dict of contrasts and list of design matrices and generates
    a dict of contrast titles & HTML for SVG Image data url
    for corresponding contrast plot.
    
    Parameters
    ----------
    contrasts: Dict[str, np.array or str]
        Contrast information, as a dict
          {'contrast_title_1, contrast_info_1/title_1, ...}
    
    design_matrices: List[pd.Dataframe]
        Design matrices computed in the model.

    Returns
    -------
    contrast_plots: Dict[str, svg img]
        Dict of contrast title and svg image data url
        for corresponding contrast plot.
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
            url_contrast_plot_svg = plot_to_svg(contrast_plot)
            contrasts_for_subsitution = {
                'contrast_plot': url_contrast_plot_svg
                }
            contrast_text_ = contrast_text_.safe_substitute(
                    contrasts_for_subsitution
                    )
            all_contrasts_plots[contrast_name] = contrast_text_
    return all_contrasts_plots


def _make_headings(contrasts, title):
    """ Creates report page title, heading & sub-heading
     using title text or contrast names.
    Accepts contrasts and user supplied title string.
    
    If title is not in (None, 'auto'), page title == heading, no sub-heading
    
    Parameters
    ----------
    contrasts: Dict[str, np.array or str]
        Contrast information, as a dict in the form
            {'contrast_title_1': contrast_info_1/title_1, ...}
        Contrast titles are used in page title and secondary heading
        if `title` is not 'auto' or None.
    
    title: String
        User supplied text for HTML Page title and primary heading.
        Overrides title auto-generation.
    
    Returns
    -------
    (HTML page title, heading, sub-heading): Tuple[str, str, str]
        If title is user-supplied, then subheading is empty string.
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


def _make_attributes_table(model):
    """ Returns an HTML table with pertinent model attributes & information.
    
    Parameters
    ----------
    model: FirstLevelModel or SecondLevelModel object.
    
    Returns
    -------
    HTML Table: String
        HTML table with the pertinent attributes of the model.
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
    display_attributes = {
        attr_name: model.__dict__[attr_name]
        for attr_name in selected_attributes
        if attr_name in model.__dict__
        }
    labels_ = display_attributes['labels_']
    if len(labels_) == 1 and isinstance(labels_[0], (np.ndarray, list, tuple)):
        display_attributes['labels_'] = labels_[0]
    model_attributes_table = pd.DataFrame.from_dict(display_attributes,
                                                    orient='index',
                                                    )
    model_attributes_table = model_attributes_table.to_html(header=False,
                                                            sparsify=False,
                                                            )
    return model_attributes_table


def make_stat_maps(model, contrasts):
    """ Given a model and contrasts, return the corresponding z-maps
    
    Parameters
    ----------
    model: FirstLevelModel or SecondLevelModel object
        Must have a fitted design matrix(ces).
    
    contrasts: Dict[str, ndarray or str]
        Dict of contrasts for a first or second level model.
        Corresponds to the contrast_def for the FirstLevelModel [1]_
        & second_level_contrast for a SecondLevelModel [2]_ .

    
    Returns
    -------
    statistical_maps: Dict[str, niimg]
        Dict of statistical z-maps keyed to contrast names/titles.
        
    See Also
    --------
    .. [1] nistats.first_level_model.FirstLevelModel.compute_contrast
    .. [2] nistats.second_level_model.SecondLevelModel.compute_contrast
    """
    statistical_maps = {contrast_id: model.compute_contrast(contrast_val)
                        for contrast_id, contrast_val in contrasts.items()
                        }
    return statistical_maps


def _make_svg_url_design_matrices(design_matrices):
    """ Accepts a FirstLevelModel or SecondLevelModel object
    with fitted design matrices & generates SVG Image URL,
    which can be inserted into an HTML template.
    
    Parameters
    ----------
    design_matrices: List[pd.Dataframe]
        Design matrices computed in the model.
        
    Returns
    -------
    svg_url_design_matrices: String
        SVG Image URL for the plotted design matrices,
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
        url_design_matrix_svg = plot_to_svg(dmtx_plot)
        dmtx_text_ = dmtx_text_.safe_substitute(
                {'design_matrix': url_design_matrix_svg}
                )
        html_design_matrices.append(dmtx_text_)
    svg_url_design_matrices = ''.join(html_design_matrices)
    return svg_url_design_matrices


def plot_to_svg(plot):
    """
    Creates an SVG image as a data URL
    from a Matplotlib Axes or Figure object.
    
    Parameters
    ----------
    plot: Matplotlib Axes or Figure object
        Contains the plot information.

    Returns
    -------
    url_plot_svg: String
        SVG Image Data URL
    """
    with io.BytesIO() as buffer:
        try:
            plot.figure.savefig(buffer, format='svg')
        except AttributeError:
            plot.savefig(buffer, format='svg')
        plot_svg = buffer.getvalue()
    url_plot_svg = quote(plot_svg.decode('utf8'))
    return url_plot_svg


def _make_roi_svg(roi_img, bg_img):
    """
    Plot cuts of an ROI/mask image and creates SVG code of it.
    
    Parameters
    ----------
    roi_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The ROI/mask image, it could be binary mask or an atlas or ROIs
        with integer values.

    bg_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image that the ROI/mask will be plotted on top of.
        To turn off background image, just pass "bg_img=None".

    Returns
    -------
    roi_plot_svg: str
        SVG Image Data URL for the ROI plot.
    """
    if roi_img:
        roi_plot = plot_roi(roi_img=roi_img, bg_img=bg_img)
        roi_plot_svg = plot_to_svg(plt.gcf())
    else:
        roi_plot_svg = None  # HTML image tag's alt attribute is used.
    return roi_plot_svg


def _make_report_components(stat_img, contrasts_plots, threshold,
                            alpha,
                            cluster_threshold, height_control, min_distance,
                            bg_img,
                            display_mode, plot_type):
    """ Populates a smaller HTML sub-template with the proper values,
     make a list containing one or more of such components
     & returns the list to be inserted into the HTML Report Template.
    Each component contains the HTML code for
    a contrast & its corresponding statistical maps & cluster table;
    
    Parameters
    ----------
    stat_img : Niimg-like object or None
       statistical image (presumably in z scale)
       whenever height_control is 'fpr' or None,
       stat_img=None is acceptable.
       If it is 'fdr' or 'bonferroni',
       an error is raised if stat_img is None.
    
    contrasts_plots: Dict[str, str]
        Contains the contrast names & the HTML code of the contrast's SVG plot.
    
    threshold: float
       desired threshold in z-scale.
       This is used only if height_control is None

    
    alpha: float
        number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.
    
    bg_img : Niimg-like object
        Only used when plot_type is 'slice'.
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image that the ROI/mask will be plotted on top of.
        If nothing is specified, the MNI152 template will be used.
        To turn off background image, just pass "bg_img=False".
    
    display_mode: string
        Choose the direction of the cuts:
        'x' - sagittal, 'y' - coronal, 'z' - axial,
        'l' - sagittal left hemisphere only,
        'r' - sagittal right hemisphere only,
        'ortho' - three cuts are performed in orthogonal directions.
        
        Possible values are:
        'ortho', 'x', 'y', 'z', 'xz', 'yx', 'yz',
        'l', 'r', 'lr', 'lzr', 'lyr', 'lzry', 'lyrz'.

    plot_type: string
        ['slice', 'glass']
        The type of plot to be drawn.
    
    Returns
    -------
    all_components: List[String]
        Each element is a set of HTML code for
        contrast name, contrast plot, statistical map, cluster table.
    """
    all_components = []
    components_template_path = os.path.join(html_template_root_path,
                                            'report_components_template.html'
                                            )
    with open(components_template_path) as html_template_obj:
        components_template_text = html_template_obj.read()
    for contrast_name, stat_map_img in stat_img.items():
        component_text_ = string.Template(components_template_text)
        stat_map_html_code = _make_stat_map_svg(
                stat_img=stat_map_img,
                threshold=threshold,
                alpha=alpha,
                cluster_threshold=cluster_threshold,
                height_control=height_control,
                bg_img=bg_img,
                display_mode=display_mode,
                plot_type=plot_type,
                )
        cluster_table_details_html, cluster_table_html = (
            _make_cluster_table_html(statistical_map_img=stat_map_img,
                                     stat_threshold=threshold,
                                     cluster_threshold=cluster_threshold,
                                     alpha=alpha,
                                     height_control=height_control,
                                     min_distance=min_distance)
        )
        components_values = {
            'contrast_name': escape(contrast_name),
            'contrast_plot': contrasts_plots[contrast_name],
            'stat_map_img': stat_map_html_code,
            'cluster_table_details': cluster_table_details_html,
            'cluster_table': cluster_table_html,
            }
        component_text_ = component_text_.safe_substitute(**components_values)
        all_components.append(component_text_)
    return all_components


def _make_stat_map_svg(stat_img,
                       threshold,
                       alpha,
                       cluster_threshold,
                       height_control,
                       bg_img,
                       display_mode,
                       plot_type,
                       ):
    """ Generates SVG code for a statistical map.
    
    Parameters
    ----------
    stat_img : Niimg-like object or None
       statistical image (presumably in z scale)
       whenever height_control is 'fpr' or None,
       stat_img=None is acceptable.
       If it is 'fdr' or 'bonferroni',
       an error is raised if stat_img is None.
       
    threshold: float
       desired threshold in z-scale.
       This is used only if height_control is None

    alpha: float
        number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.

    cluster_threshold : float
        cluster size threshold. In the returned thresholded map,
        sets of connected voxels (`clusters`) with size smaller
        than this number will be removed.

    height_control: string
        false positive control meaning of cluster forming
        threshold: 'fpr'|'fdr'|'bonferroni'|None

    bg_img : Niimg-like object
        Only used when plot_type is 'slice'.
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image that the ROI/mask will be plotted on top of.
        If nothing is specified, the MNI152 template will be used.
        To turn off background image, just pass "bg_img=False".
    
    display_mode: string
        Choose the direction of the cuts:
        'x' - sagittal, 'y' - coronal, 'z' - axial,
        'l' - sagittal left hemisphere only,
        'r' - sagittal right hemisphere only,
        'ortho' - three cuts are performed in orthogonal directions.
        
        Possible values are:
        'ortho', 'x', 'y', 'z', 'xz', 'yx', 'yz',
        'l', 'r', 'lr', 'lzr', 'lyr', 'lzry', 'lyrz'.
    
    plot_type: string
        ['slice', 'glass']
        The type of plot to be drawn.
        
    
    Returns
    -------
    stat_map_svg: string
        SVG Image Data URL representing a statistical map.
    """
    thresholded_stat_map, _ = map_threshold(stat_img,
                                                threshold=threshold,
                                                alpha=alpha,
                                                cluster_threshold=cluster_threshold,
                                                height_control=height_control,
                                                )
    if plot_type == 'slice':
        stat_map_plot = plot_stat_map(thresholded_stat_map,
                                      bg_img=bg_img,
                                      display_mode=display_mode,
                                      )
    elif plot_type == 'glass':
        stat_map_plot = plot_glass_brain(thresholded_stat_map,
                                         display_mode=display_mode,
                                         colorbar=True,
                                         plot_abs=False,
                                         )
    else:
        raise ValueError('Invalid plot type provided. Acceptable options are\n'
                         "'slice' or 'glass'.")
    
    stat_map_svg = plot_to_svg(plt.gcf())
    return stat_map_svg


def _make_cluster_table_html(statistical_map_img,
                             stat_threshold,
                             cluster_threshold,
                             alpha,
                             height_control,
                             min_distance,
                             ):
    """ Makes a HTML tables for clustering details & a cluster table.

    Parameters
    ----------
    stat_img : Niimg-like object,
       Statistical image (presumably in z- or p-scale).

    stat_threshold: `float`
        Cluster forming threshold in same scale as `stat_img` (either a
        p-value or z-scale value).

    cluster_threshold : `int` or `None`, optional
        Cluster size threshold, in voxels.

    alpha: float
        For display purposes only.
        Number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.
        
    height_control: string
        For display purposes only.
        false positive control meaning of cluster forming
        threshold: 'fpr'|'fdr'|'bonferroni'|None
    
    min_distance: `float`
        For display purposes only.
        Minimum distance between subpeaks in mm. Default is 8 mm.

    Returns
    -------
    table_details_html: String
        HTML table with clustering details
    cluster_table_html: String
        HTML table with clusters.
    """
    cluster_table = get_clusters_table(statistical_map_img,
                                       stat_threshold=stat_threshold,
                                       cluster_threshold=cluster_threshold,
                                       min_distance=min_distance,
                                       )
    table_details = OrderedDict()
    table_details.update({'Threshold Z': stat_threshold})
    table_details.update({'Cluster size threshold (voxels)':
                                      cluster_threshold
                                  }
                                 )
    table_details.update({'Minimum distance (mm)': min_distance})
    table_details.update({'Height control': height_control})
    table_details.update({'Cluster Level p-value Threshold': alpha})
    try:
        table_details_html = pd.DataFrame.from_dict(
                table_details, orient='index').to_html(border=0,
                                                               header=False,
                                                               )
    except TypeError:  # pandas < 0.19
        table_details_html = pd.DataFrame.from_dict(
                table_details, orient='index').to_html(header=False,
                                                               )
    with pd.option_context('display.precision', 2):
        cluster_table_html = cluster_table.to_html()
    return table_details_html, cluster_table_html
