import string
import os
import pprint

import numpy as np

from nistats.design_matrix import check_design_matrix


html_template_path = os.path.join(os.path.dirname(__file__), 'report_template.html')
def generate_report(output_path, model, **kwargs):
    with open(html_template_path) as html_file_obj:
        html_template_text = html_file_obj.read()
        
    # print(html_template_text)
    report_template = string.Template(html_template_text)
    contrasts = pretty_print_mapping_of_sequence(kwargs['contrasts'])
    print(repr(contrasts))
    design_matrix_plot = save_design_matrix_plot(model)
    report_values = {'Title': 'Test Report',
                     'contrasts': contrasts,
                     'design_matrix_binary': design_matrix_plot,
                     }
    report_text = report_template.safe_substitute(**report_values)
    
    print(report_text)
    with open(output_path, 'w') as html_write_obj:
        html_write_obj.write(report_text)


def save_design_matrix_plot(model):
    from matplotlib import pyplot as plt
    
    design_matrix = model.design_matrices_[0]
    fig = plt.figure()  #TODO: code from _plot_matrices.plot_design_matrix(). Refactor target.
    ax = fig.add_subplot(1, 1, 1)
    _, X, names = check_design_matrix(design_matrix)
    X = X / np.maximum(1.e-12, np.sqrt(
            np.sum(X ** 2, 0)))  # pylint: disable=no-member

    ax.imshow(X, interpolation='nearest', aspect='auto')
    ax.set_label('conditions')
    ax.set_ylabel('scan number')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha='right')

    plt.tight_layout()
    
    # dmtx_fpath = os.path.abspath('./results/dmtx.png')
    dmtx_fpath = 'dmtx.png'
    fig.savefig(dmtx_fpath)
    return dmtx_fpath
    
    # experimenting with embedding the image binary into the html
    # fig.canvas.draw()
    # buf = fig.canvas.tostring_rgb()
    # import base64
    # buf64 = base64.b64encode(buf)
    # design_matrix_image_html_insert = 'data:image/png;base64,{}'.format((buf64))
    # design_matrix_image_html_insert = '<img src="data:image/png;base64,{}">'.format(buf64.rstrip('\r'))
    # return design_matrix_image_html_insert
    
    
def pretty_print_mapping_of_sequence(any_dict):
    output_text = []
    for key, value in any_dict.items():
        formatted_value = pprint.pformat(value)
        line_text = '{} : {}'.format(key, formatted_value)
        output_text.append(line_text)
    return '\n'.join(output_text)
        
    
if __name__ == '__main__':
    generate_report('generated_report.html', None)
