import string
import os

import pandas as pd
from nilearn.datasets import load_mni152_template
from nilearn.plotting import plot_stat_map, plot_glass_brain

from nistats.reporting import (
    plot_design_matrix,
    get_clusters_table,
    )


html_template_root_path = os.path.dirname(__file__)


def make_glm_report(output_path,
                    model,
                    contrasts,
                    title='auto',
                    threshold=3.09,
                    bg_img='MNI 152 Template',
                    display_mode='z',
                    plot_type='stat',
                    ):
    """ Creates an HTML page which shows all important aspects
    of a fitted GLM.
    
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
        
    Returns
    -------
    None
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
    # print(report_text)
    from nilearn.plotting.js_plotting_utils import HTMLDocument
    # return HTMLDocument(report_text)
    with open(output_path, 'w') as html_write_obj:
        html_write_obj.write(report_text)


def _make_contrasts_dict(contrasts):
    # contrasts = 'StopSuccess - Go'
    contrasts = [contrasts] if isinstance(contrasts, str) else contrasts
    if not isinstance(contrasts, dict):
        contrasts = {contrast_: contrast_ for contrast_ in contrasts}
    return contrasts


def _make_page_title_heading(contrasts, title):
    if title != 'auto':
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
    """ Given a model and contrasts, return the corresponding z-maps"""
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
    String of HTML code to be inserted into the HTML template
    to insert the plotted design matrices.
    """
    html_design_matrices = []
    for count, design_matrix in enumerate(model.design_matrices_):
        dmtx_filepath = 'dmtx{}.svg'.format(count)
        dmtx = plot_design_matrix(design_matrix)
        import io
        buffer = io.StringIO()
        dmtx.figure.savefig(buffer, format='svg')
            
        # html_design_matrix = ('<img src="{}" alt="Visual representation '
        #                       'of Design Matrix of the fMRI experiment">'
        #                       ).format(dmtx_filepath)
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
    String of HTML code representing Statistical Maps + Cluster Tables
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
        stat_map_plot_filepath = _make_html_for_stat_maps(stat_map_name,
                                                          stat_map_img,
                                                          threshold,
                                                          bg_img,
                                                          display_mode,
                                                          plot_type ,
                                                          )
        cluster_table_html = _make_html_for_cluster_table(stat_map_img, threshold)
        components_values = {
            'contrast': contrast_html,
            'stat_map_img': stat_map_plot_filepath,
            'cluster_table': cluster_table_html,
            }
        component_text_ = component_text_.safe_substitute(**components_values)
        all_components.append(component_text_)
    return all_components


def _make_html_for_contrast(stat_map_name, contrasts):
    contrast_val = contrasts[stat_map_name]
    contrast_val = '' if isinstance(contrast_val, str) else contrast_val
    current_contrast = {stat_map_name: contrast_val}
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
    z_map_name_filename_text = statistical_map_name.title().replace(' ', '')
    stat_map_plot_filepath = 'stat_map_plot_{}.svg'.format(
            z_map_name_filename_text)
    stat_map_plot.savefig(stat_map_plot_filepath)
    return stat_map_plot_filepath


def _make_html_for_cluster_table(statistical_map_img, threshold):
    """ Generates string of HTML code for a cluster table.

    Parameters
    ----------
    statistical_map_img: Nifti image

    Returns
    -------
    String of HTML code representing a cluster table.
    """
    cluster_table = get_clusters_table(statistical_map_img,
                                       stat_threshold=threshold,
                                       cluster_threshold=15,
                                       )
    single_cluster_table_html_code = cluster_table.to_html()
    return single_cluster_table_html_code


# def save_design_matrix_plot(model):
#     """
#     :param model: not sure if I will use this, or stick with plot_design_matrix
#     :return:
#     """
#     from matplotlib import pyplot as plt
#
#     design_matrix = model.design_matrices_[0]
#     fig = plt.figure()  #TODO: code from _plot_matrices.plot_design_matrix(). Refactor target.
#     ax = fig.add_subplot(1, 1, 1)
#     _, X, names = check_design_matrix(design_matrix)
#     X = X / np.maximum(1.e-12, np.sqrt(
#             np.sum(X ** 2, 0)))  # pylint: disable=no-member
#
#     ax.imshow(X, interpolation='nearest', aspect='auto')
#     ax.set_label('conditions')
#     ax.set_ylabel('scan number')
#
#     ax.set_xticks(range(len(names)))
#     ax.set_xticklabels(names, rotation=60, ha='right')
#
#     plt.tight_layout()
#
#     # dmtx_fpath = os.path.abspath('./results/dmtx.png')
#     dmtx_fpath = 'dmtx.png'
#     fig.savefig(dmtx_fpath)
#     return dmtx_fpath

#   # experimenting with embedding the image binary into the html
#   # fig.canvas.draw()
#   # buf = fig.canvas.tostring_rgb()
#   # import base64
#   # buf64 = base64.b64encode(buf)
#   # design_matrix_image_html_insert = 'data:image/png;base64,{}'.format((buf64))
#   # design_matrix_image_html_insert = '<img src="data:image/png;base64,{}">'.format(buf64.rstrip('\r'))
#   # return design_matrix_image_html_insert
    
   
    

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
