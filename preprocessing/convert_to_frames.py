import av
import os
import cv2

from pathlib import Path

data_folder = Path("/Users/andreybezumennui/Downloads/Normal_Videos_for_Event_Recognition")
# find . -name '.DS_Store' -type f -delete


def read_write_with_cv(file_path):
    vc = cv2.VideoCapture(str(file_path))
    c = 1
    save_path_dir = str(file_path)[:-4]
    os.system(f'mkdir {save_path_dir}')
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        try:
            rval, frame = vc.read()
            cv2.imwrite(f'{save_path_dir}/frame-%04d.jpg' % c, frame)
            c = c + 1
            cv2.waitKey(1)
        except Exception as e:
            print(e)
    vc.release()


def read_write_with_av(file_path):
    container = av.open(str(file_path))
    save_path_dir = str(file_path)[:-4]
    os.system(f'mkdir {save_path_dir}')
    for packet in container.demux():
        for frame in packet.decode():
            # if frame.type == 'video':
            frame.to_image().save(f'{save_path_dir}/frame-%04d.png' % frame.index)


def processing():
    dirs = os.listdir(data_folder)
    for dir in dirs:
        for file in os.listdir(data_folder / dir):
            # read_write_with_av(data_folder / f'{dir}/{file}')
            print(data_folder / f'{dir}/{file}')
            read_write_with_cv(data_folder / f'{dir}/{file}')


if __name__ == '__main__':
    processing()

