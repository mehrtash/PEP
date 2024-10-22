import os
# todo: change the following path to the desired project root
project_root = '.'

folders = list()

raw_folder = os.path.join(project_root, 'raw')
folders.append(raw_folder)

intermediate_folder = os.path.join(project_root, 'intermediate')
folders.append(intermediate_folder)

models_folder = os.path.join(intermediate_folder, 'models')
folders.append(models_folder)

data_folder = os.path.join(intermediate_folder, 'data')
folders.append(data_folder)

arrays_folder = os.path.join(data_folder, 'arrays')
folders.append(arrays_folder)

sheets_folder = os.path.join(data_folder, 'sheets')
folders.append(sheets_folder)

if __name__ == '__main__':
    for folder in folders:
        if not os.path.isdir(folder):
            os.mkdir(folder)
