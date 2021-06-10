import os
import argparse

def del_tags(tags, load_file, save_file,right_pad_space=False, left_pad_space=False):
    if right_pad_space:
        tags = [tag + " " for tag in tags]
    if left_pad_space:
        tags = [" " + tag for tag in tags]

    with open(load_file, "r") as f2:
        lines = f2.readlines()
    f2.close()


    with open (save_file,"w") as f1:
        print(tags)
        for line in lines:
            for tag in tags:
                line = line.replace(tag, "")
                line = line.replace(" .", ".")
                line = line.replace(" ,", ",")
                f1.write(line)
        print(line)
    f1.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_file", help="load file")
    parser.add_argument("--save_file", help="save file")
    parser.add_argument("--tags_to_del", help="tags to del, seperated by space.")
    args = parser.parse_args()

    tags_to_del = args.tags_to_del.split()
    with open("/home/ubuntu/efs-storage/dataset/webnlg/test.txt", "r") as f:
        L = f.readlines()
    del_tags(tags_to_del, args.load_file, args.save_file, right_pad_space=True)