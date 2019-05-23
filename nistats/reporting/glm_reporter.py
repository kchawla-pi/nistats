import string

# template_text = (
#     '<!DOCTYPE html>\n'
#     '<html lang="en">\n'
#     '<head>\n'
#     '<meta charset="UTF-8">\n'
#     '<title>Title</title>\n'
#     '</head>\n'
#     '<body>\n'
#
# </body>
# </html>
# '
# )
with open('report_template.html') as html_file_obj:
    html_template_text = html_file_obj.read()
    
# print(html_template_text)
report_template = string.Template(html_template_text)

report_values = {'Title': 'Test Report'}
report_text = report_template.safe_substitute(**report_values)
print(report_text)
with open('generated_report.html', 'w') as html_write_obj:
    html_write_obj.write(report_text)
