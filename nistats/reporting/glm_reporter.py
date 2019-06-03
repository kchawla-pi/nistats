import string
import os
import pprint

import numpy as np
from nilearn.plotting import plot_stat_map
from nistats.reporting import (
    plot_design_matrix,
    get_clusters_table,
    )


html_template_path = os.path.join(os.path.dirname(__file__), 'report_template.html')


def _report_design_matrices(model):
    html_design_matrices = []
    for count, design_matrix in enumerate(model.design_matrices_):
        dmtx_filepath = 'dmtx{}.png'.format(count)
        plot_design_matrix(design_matrix, output_file=dmtx_filepath)
        html_design_matrix = '''<img src="{}" alt="Visual representation of Design Matrix of the fMRI experiment">
    '''.format(dmtx_filepath)
        html_design_matrices.append(html_design_matrix)
        # design_matrix_plot = save_design_matrix_plot(model)
    html_design_matrices = '\n'.join(html_design_matrices)
    return html_design_matrices


def _report_stat_maps(z_maps, bg_img):
    pass


def generate_report(output_path, model, **kwargs):
    # check_glm_model_attr(model)
    with open(html_template_path) as html_file_obj:
        html_template_text = html_file_obj.read()
        
    # print(html_template_text)
    report_template = string.Template(html_template_text)
    contrasts_display_text = pretty_print_mapping_of_sequence(kwargs['contrasts'])
    
    html_design_matrices = _report_design_matrices(model)
    
    z_maps = kwargs['z_maps']
    anatomical_img = kwargs['bg_img']
    html_stat_maps = []
    html_code_for_cluster_tables = []
    # html_code_stat_map_cluster_table_pair = []
    for z_map_name in z_maps:
    # z_map_name = list(z_maps.keys())[0]
        stat_map_plot = plot_stat_map(z_maps[z_map_name],
                                      threshold=kwargs['threshold'],
                                      title=z_map_name,
                                      bg_img=anatomical_img,
                                      display_mode=kwargs['display_mode'],
                                      )
        z_map_name_filename_text = z_map_name.title().replace(' ', '')
        stat_map_plot_filepath = 'stat_map_plot_{}.png'.format(z_map_name_filename_text )
        stat_map_plot.savefig(stat_map_plot_filepath)
        html_stat_map_ = '''<img src="{}">'''.format(stat_map_plot_filepath)
        html_stat_maps.append(html_stat_map_)
        cluster_table = get_clusters_table(z_maps[z_map_name], 3.09, 15)
        cluster_table_html = cluster_table.to_html()
        html_code_for_cluster_table_ = '''<p>{}</p>'''.format(cluster_table_html)
        html_code_for_cluster_tables.append(html_code_for_cluster_table_)
    html_stat_maps = '\n'.join(html_stat_maps)
    html_code_for_cluster_tables = '\n'.join(html_code_for_cluster_tables)
    report_values = {'Title': 'Test Report',
                     'contrasts': contrasts_display_text,
                     'design_matrix_binary': html_design_matrices,
                     'html_stat_maps': html_stat_maps,
                     'html_code_for_cluster_tables': html_code_for_cluster_tables,
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
    generate_report('generated_report.html', None)
