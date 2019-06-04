import string
import os
import pprint

import numpy as np
from nilearn.datasets import load_mni152_template
from nilearn.plotting import plot_stat_map
from nistats.reporting import (
    plot_design_matrix,
    get_clusters_table,
    )


html_template_path = os.path.join(os.path.dirname(__file__), 'report_template.html')


def make_report(output_path,
                model,
                contrasts,
                statistical_maps,
                threshold=3.09,
                bg_img='MNI 152 Template',
                display_mode='z'):
    """ Creates an HTML page which shows all important aspects
    of a fitted GLM.
    
    Parameters
    ----------
    output_path: String (Path)
        The file path to which the HTML report will be saved.
    
    model: FirstLevelModel or SecondLevelModel object
        A fitted first or second level model object.
        
    contrasts: Dict[string, ndarray]
    
    statistical_maps: Dict[string, ndarray]
    
    threshold: float
        Default is 3.09
        
    bg_img: Niimg
        Default is the MNI152 template
        
    display_mode: String
        Default is 'z'.
        
    Returns
    -------
    None
    """
    bg_img = load_mni152_template() if bg_img == 'MNI 152 Template' else bg_img
    
    with open(html_template_path) as html_file_obj:
        html_template_text = html_file_obj.read()
    
    # print(html_template_text)
    report_template = string.Template(html_template_text)
    contrasts_display_text = pretty_print_mapping_of_sequence(contrasts)
    
    html_design_matrices = _report_design_matrices(model)
    all_stat_map_cluster_table_pairs_html_code = (
        _report_stat_maps_cluster_tables(statistical_maps,
                                         threshold,
                                         bg_img,
                                         display_mode,
                                         )
    )
    
    report_values = {'Title': 'Test Report',
                     'contrasts': contrasts_display_text,
                     'design_matrices': html_design_matrices,
                     'all_stat_map_cluster_table_pairs': all_stat_map_cluster_table_pairs_html_code,
                     }
    report_text = report_template.safe_substitute(**report_values)
    print(report_text)
    with open(output_path, 'w') as html_write_obj:
        html_write_obj.write(report_text)


def pretty_print_mapping_of_sequence(any_dict):
    output_text = []
    for key, value in any_dict.items():
        formatted_value = pprint.pformat(value)
        line_text = '{} : {}'.format(key, formatted_value)
        output_text.append(line_text)
    return '\n'.join(output_text)


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
        dmtx_filepath = 'dmtx{}.png'.format(count)
        plot_design_matrix(design_matrix, output_file=dmtx_filepath)
        html_design_matrix = ('<img src="{}" alt="Visual representation '
                              'of Design Matrix of the fMRI experiment">'
                              ).format(dmtx_filepath)
        html_design_matrices.append(html_design_matrix)
    html_design_matrices = '\n'.join(html_design_matrices)
    return html_design_matrices


def _report_stat_maps_cluster_tables(statistical_maps, threshold, bg_img, display_mode):
    """ Generates a string of HTML code
    representing statistical maps & corresponding cluster tables;
    to be inserted into the HTML Report Template.
    
    Parameters
    ----------
    statistical_maps: Dict[string, numpy array]
    
    threshold: float
    
    bg_img: niimg
    
    display_mode: string
    
    Returns
    -------
    String of HTML code representing Statistical Maps + Cluster Tables
    """
    all_stat_map_cluster_table_pairs_html_code = []
    for stat_map_name, stat_map_data in statistical_maps.items():
        single_stat_map_html_code = _make_html_for_stat_maps(stat_map_name,
                                                             stat_map_data,
                                                             threshold,
                                                             bg_img,
                                                             display_mode,
                                                             )
        single_cluster_table_html_code = _make_html_for_cluster_table(stat_map_data)
        single_stat_map_cluster_table_pair = [single_stat_map_html_code,
                                              single_cluster_table_html_code]
        all_stat_map_cluster_table_pairs_html_code.extend(
            single_stat_map_cluster_table_pair)
    all_stat_map_cluster_table_pairs_html_code = '\n'.join(
        all_stat_map_cluster_table_pairs_html_code)
    return all_stat_map_cluster_table_pairs_html_code


def _make_html_for_stat_maps(statistical_map_name,
                             statistical_map_data,
                             threshold,
                             bg_img,
                             display_mode,
                             ):
    """ Generates string of HTML code for a statistical map.
    
    Parameters
    ----------
    statistical_map_name: String
    
    statistical_map_data: Ndarray
    
    threshold: float
    
    bg_img: niimg
    
    display_mode: String
    
    Returns
    -------
    String of HTML code representing a statistical map.
    """
    stat_map_plot = plot_stat_map(statistical_map_data,
                                  threshold=threshold,
                                  title=statistical_map_name,
                                  bg_img=bg_img,
                                  display_mode=display_mode,
                                  )
    z_map_name_filename_text = statistical_map_name.title().replace(' ', '')
    stat_map_plot_filepath = 'stat_map_plot_{}.png'.format(
            z_map_name_filename_text)
    stat_map_plot.savefig(stat_map_plot_filepath)
    single_stat_map_html_code = '''<img src="{}">'''.format(
            stat_map_plot_filepath)
    return single_stat_map_html_code


def _make_html_for_cluster_table(statistical_map_data):
    """ Generates string of HTML code for a cluster table.

    Parameters
    ----------
    statistical_map_data: Ndarray

    Returns
    -------
    String of HTML code representing a cluster table.
    """
    cluster_table = get_clusters_table(statistical_map_data, 3.09, 15)
    single_cluster_table_html_code = cluster_table.to_html()
    single_cluster_table_html_code = '''<p>{}</p>'''.format(
            single_cluster_table_html_code)
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
    make_report('generated_report.html', None)
