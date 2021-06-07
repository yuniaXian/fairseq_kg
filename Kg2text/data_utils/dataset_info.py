import argparse, os, json
import itertools, math

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="", help="file name")
    parser.add_argument("--data_dir", type=str, default="/home/xianjiay/efs-storage/data-bin/dataset", help="data_dir")
    parser.add_argument("--dataset", type=str, default="webnlg", help="specify dataset: webnlg, kgtext_wikidata, wikibionlg")
    parser.add_argument("--option", type=str, default="info", help="info/slice/merge")
    parser.add_argument("--unit", type=int, default=0, help="num of lines to be written in one seperate file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.option == "info":
        file_name_abs = os.path.join(args.data_dir, args.dataset, args.file_name)
        if file_name_abs.endswith(".json"):
            with open(file_name_abs, 'r') as f:
                data = json.load(f)
        else:
            with open(file_name_abs, 'r') as f:
                data = f.readlines()

        f.close()
        print("Done")

    elif args.option == "slice":
        file_name_abs = os.path.join(args.data_dir, args.dataset, args.file_name)

        if file_name_abs.endswith(".json"):
            with open(file_name_abs, 'r') as f:
                data = json.load(f)
        else:
            with open(file_name_abs, 'r') as f:
                data = f.readlines()
        f.close()

        size_data = len(data)
        assert args.unit <= size_data
        suffix = "." + file_name_abs.split(".")[-1]
        L = len(suffix)
        path = file_name_abs[:-L]
        slices = math.ceil(size_data/args.unit)
        
        if suffix == ".json":

            for k in range(slices):
                path_k = path + str(k) + suffix
                s, t = k*args.unit, max((k+1)*args.unit, size_data)
                with open(path_k, "w") as fw:
                    json.dump(data[s:t], fw)
                fw.close()
        
        else:
            for k in range(slices):
                path_k = path + str(k) + suffix
                s, t = k*args.unit, max((k+1)*args.unit, size_data)
                with open(path_k, "w") as fw:
                    for element in data[s:t]:
                        fw.write(element + "\n")
                fw.close()

    elif args.option == "merge":
        pass