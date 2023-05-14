import os
import shutil
import glob


def main():
    list_folders = sorted(glob.glob('results/**/**'))

    for folder in list_folders:
        data_folder = os.path.join(folder, 'data.pickle')

        if not os.path.exists(data_folder):
            shutil.rmtree(folder)
            print(f'{folder} deleted!')


if __name__ == '__main__':
    main()
