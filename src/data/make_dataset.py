from os import path, replace, scandir

"""
This script is to format the raw data into proper formats: jpg, csv 
jpg is the image file
csv is the file containing the box cordinates
"""

def noext(f):
    return path.splitext(f.name)[0]

raw_task1_files = list(
    sorted(scandir("0325updated.task1train(626p)"), key=lambda f: f.name)
)
raw_task2_files = list(
    sorted(scandir("0325updated.task2train(626p)"), key=lambda f: f.name)
)

jpg_files = [f for f in raw_task1_files if f.name.endswith("jpg")]
csv_files = [f for f in raw_task1_files if f.name.endswith("txt")]


for i, (f_jpg, f_csv) in enumerate(zip(jpg_files, csv_files)):
    if noext(f_jpg) != noext(f_csv) or noext(f_csv):
        raise ValueError("Raw data filenames mismatch")

    print(f"{i:03d}", f_jpg, f_csv)

    replace(f_jpg.path, f"data/raw/img{i:03d}.jpg")
    replace(f_csv.path, f"data/raw/box{i:03d}.csv")
