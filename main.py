import papermill as pm
import os

# List files in the data directory
file_list = os.listdir("data")
# List arbitrary names for the generated reports
name_list = ["long_signal1", "long_signal2", "short_signal"]

# For every data file, execute analysis and save report
for name, file in zip(name_list, file_list):

    res = pm.execute_notebook(
        'report.ipynb',
        'reports/{name}.ipynb',
        parameters = dict(file=file, name=name)
    )


# Store reports dir into a variable
report_dir = "reports/"
# List notebooks in reports dir
report_list = os.listdir(report_dir)

# Convert each notebook to markdown
for report in report_list:

    # If file contains ipynb, then it's a notebook to be converted
    if "ipynb" in report:
        print(report_dir + report, "is being processed.")

        # Define destination file
        destination = report_dir + report
        # Define shell command for conversion
        exe_command = f'jupyter nbconvert "{destination}" --to markdown'

        # Run command
        os.system(exe_command)
        # Remove initial notebook
        # os.remove(report_dir + report)

