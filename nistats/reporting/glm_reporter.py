import io
import string
import os

import pandas as pd

from matplotlib import pyplot as plt
from nilearn.datasets import load_mni152_template
from nilearn.plotting import plot_stat_map, plot_glass_brain
from nilearn.plotting.js_plotting_utils import HTMLDocument

from nistats.reporting import (
    plot_design_matrix,
    get_clusters_table,
    )


html_template_root_path = os.path.dirname(__file__)


def make_glm_report(
                    model,
                    contrasts,
                    title='auto',
                    threshold=3.09,
                    bg_img='MNI 152 Template',
                    display_mode='z',
                    plot_type='stat',
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
    output_path: String (Path)
        The file path to which the HTML report will be saved.
    
    model: FirstLevelModel or SecondLevelModel object
        A fitted first or second level model object.
        
    contrasts: Dict[string, ndarray] , String
    
    threshold: float
        Default is 3.09
        
    bg_img: Nifti image
        Default is the MNI152 template
        
    display_mode: String
        Default is 'z'.
        
    plot_type: String. ['stat' (default)| 'glass']
        Specifies the type of plot to be drawn for the statistical maps.
        
        
    Returns
    -------
    report_text: HTMLDocument Object
        Contains the HTML code for the GLM Report.
    """
    pd.set_option('display.max_colwidth', -1)
    bg_img = load_mni152_template() if bg_img == 'MNI 152 Template' else bg_img
    html_template_path = os.path.join(html_template_root_path,
                                      'report_template.html')
    with open(html_template_path) as html_file_obj:
        html_template_text = html_file_obj.read()
    contrasts = _make_contrasts_dict(contrasts)
    page_title, page_heading_1, page_heading_2 = _make_page_title_heading(contrasts,
                                                                          title,
                                                                          )
    report_template = string.Template(html_template_text)
    contrast_arrays_not_supplied = any([contrast_name == contrast_value
                                        for contrast_name, contrast_value
                                        in contrasts.items()
                                        ]
                                       )
    if contrast_arrays_not_supplied:
        contrasts_display_text = ', '.join(sorted(contrasts))
    else:
        contrasts_display_text = pd.DataFrame.from_dict(contrasts,
                                                        orient='index'
                                                        ).to_html(border=0,
                                                                  header=False,
                                                                  )
    pd.set_option('display.max_colwidth', 50)
    model_attributes_html = _make_model_attributes_html_table(model)
    statistical_maps = make_statistical_maps(model, contrasts)
    html_design_matrices = _report_design_matrices(model)
    all_components = _make_report_components(statistical_maps,
                                             contrasts,
                                             threshold,
                                             bg_img,
                                             display_mode,
                                             plot_type,
                                             )
    all_components_text = '\n'.join(all_components)
    report_values = {'page_title': page_title,
                     'page_heading_1': page_heading_1,
                     'page_heading_2': page_heading_2,
                     'model_attributes': model_attributes_html,
                     'contrasts': contrasts_display_text,
                     'design_matrices': html_design_matrices,
                     'component': all_components_text,
                     }
    report_text = report_template.safe_substitute(**report_values)
    return HTMLDocument(report_text)


def _make_contrasts_dict(contrasts):
    """ Accepts contrasts and returns a dict of them.
    with the names of contrasts as keys.
    If contrasts is a:
     dict then returns it unchanged.
     string or list/tuple of strings, returns a dict where key==values.
    
    Parameters
    ----------
    contrasts: str, list/tuple, dict
    
    Returns
    -------
    contrasts: Dict[str, np.array or str]
    """
    contrasts = [contrasts] if isinstance(contrasts, str) else contrasts
    if not isinstance(contrasts, dict):
        contrasts = {contrast_: contrast_ for contrast_ in contrasts}
    return contrasts


def _make_page_title_heading(contrasts, title):
    """ Creates report page title, heading & sub-heading
     using title text or contrast names.
    Accepts contrasts and user supplied title string.
    
    If title is not in (None, 'auto'), page title == heading, no sub-heading
    
    Parameters
    ----------
    contrasts: Dict[str, np.array or str] or List/Tuple[str] or String
        Needed for contrast names, either as a sequence or keys of dict.
    
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
    Does not contain derived attributes.
    
    Parameters
    ----------
    model: FirstLevelModel or SecondLevelModel object.
    
    Returns
    -------
    HTML Table: String
        HTML code for creating a table.
    """
    selected_model_attributes = {
        attr_name: attr_val
        for attr_name, attr_val in model.__dict__.items()
        if not attr_name.endswith('_')
        }
    model_attributes_table = pd.DataFrame.from_dict(selected_model_attributes,
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


def _report_design_matrices(model):
    """ Accepts a FirstLevelModel or SecondLevelModel object
    with fitted design matrices & generates HTML code
    to insert their plots into the report.
    
    Parameters
    ----------
    model: FirstLevelModel or SecondLevelModel object
        First or Second Level Model objects with fitted design matrices.
        
    Returns
    -------
    html_design_matrices: String
        HTML code for the plotted design matrices,
        to be inserted into the HTML template.
    """
    html_design_matrices = []
    for count, design_matrix in enumerate(model.design_matrices_):
        design_matrix_image_axes = plot_design_matrix(design_matrix)
        buffer = io.StringIO()
        design_matrix_image_axes.figure.savefig(buffer, format='svg')
        html_design_matrix = ('<svg class="dmtx">{}</svg>'.format(buffer.getvalue()))
        html_design_matrices.append(html_design_matrix)
    html_design_matrices = '\n'.join(html_design_matrices)
    return html_design_matrices


def _make_report_components(statistical_maps, contrasts, threshold, bg_img, display_mode, plot_type):
    """ Populates a smaller HTML sub-template with the proper values,
     make a list containing one or more of such components
     & returns the list to be inserted into the HTML Report Template.
    Each component contains the HTML code for
    a contrast & its corresponding statistical maps & cluster table;
    
    Parameters
    ----------
    statistical_maps: Nifti images
    
    threshold: float
    
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
    for stat_map_name, stat_map_img in statistical_maps.items():
        component_text_ = string.Template(components_template_text)
        contrast_html = _make_html_for_contrast(stat_map_name, contrasts)
        stat_map_html_code  = _make_html_for_stat_maps(stat_map_name,
                                                          stat_map_img,
                                                          threshold,
                                                          bg_img,
                                                          display_mode,
                                                          plot_type ,
                                                          )
        cluster_table_html = _make_html_for_cluster_table(stat_map_img, threshold)
        components_values = {
            'contrast': contrast_html,
            'stat_map_img': stat_map_html_code,
            'cluster_table': cluster_table_html,
            }
        component_text_ = component_text_.safe_substitute(**components_values)
        all_components.append(component_text_)
    return all_components


def _make_html_for_contrast(contrast_name, contrasts):
    """ Generates string of HTML code for a Pandas DataFrame of contrast name and array.
    Accepts statistical map name, dict of contrasts.
    
    Parameters
    ----------
    contrast_name: String
        Name of the individual contrast
        
    contrasts: Dict[str, ndarray or str]
        Dict of all contrasts.
        
    Returns
    -------
    contrast_html: String
        HTML code to insert individual contrast name, its array into a webpage.
    """
    contrast_val = contrasts[contrast_name]
    contrast_val = '' if isinstance(contrast_val, str) else contrast_val
    current_contrast = {contrast_name: contrast_val}
    contrast_html = pd.DataFrame.from_dict(current_contrast, orient='index'
                           ).to_html(header=False, border=0)
    return contrast_html
    

def _make_html_for_stat_maps(statistical_map_name,
                             statistical_map_img,
                             threshold,
                             bg_img,
                             display_mode,
                             plot_type,
                             ):
    """ Generates string of HTML code for a statistical map.
    
    Parameters
    ----------
    statistical_map_name: String
    
    statistical_map_img: Ndarray
    
    threshold: float
    
    bg_img: Nifti image
    
    display_mode: String
    
    Returns
    -------
    stat_map_html_code: String
        String of HTML code representing a statistical map.
    """
    if plot_type == 'glass':
        stat_map_plot = plot_glass_brain(statistical_map_img,
                                         threshold=threshold,
                                         title=statistical_map_name,
                                         display_mode=display_mode,
                                         )
    else:
        stat_map_plot = plot_stat_map(statistical_map_img,
                                      threshold=threshold,
                                      title=statistical_map_name,
                                      bg_img=bg_img,
                                      display_mode=display_mode,
                                      )
    buffer = io.StringIO()
    plt.savefig(buffer, format='svg')
    stat_map_html_code = buffer.getvalue()
    return stat_map_html_code


def _make_html_for_cluster_table(statistical_map_img, threshold):
    """ Generates string of HTML code for a cluster table.

    Parameters
    ----------
    statistical_map_img: Nifti image

    Returns
    -------
    single_cluster_table_html_code: String
        HTML code representing a cluster table.
    """
    cluster_table = get_clusters_table(statistical_map_img,
                                       stat_threshold=threshold,
                                       cluster_threshold=15,
                                       )
    single_cluster_table_html_code = cluster_table.to_html()
    return single_cluster_table_html_code


if __name__ == '__main__':
    make_glm_report('generated_report.html', None)

#TODO: Less peak value precision
#TODO: Add effect size of contrast?
#TODO: Limit number of voxel clusters in table
#TODO: Diagnostic things like Image of variance? Plot variance maps, effects size maps?
#TODO: Oasis VBM Example, check age effect. It is positive. It should be negative (expected reduction in cortex).
#TODO: Should we output in Markdown? Easy to cut-paste, insert in Latex.
#TODO: Plot stat maps on Glass Brains?

#TODO: Variance of the model
#TODO: Noise model
