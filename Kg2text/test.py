from numpy import append
from fairseq.data.indexed_dataset import IndexedRawTextDataset, IndexedDataset
from fairseq.data import Dictionary
import sys
import difflib


def write_to_check_file(data, file_path=None):
    if file_path is None:
        file_path = "data-bin/webnlg/sentencepiece_bped_data/check_eval.txt"
    with open(file_path, "a") as f:
        for line in data.lines:
            f.write(line + "\n")
    
    f.close()



def read_file(file_name):
    try:
        file_handle = open(file_name, 'r')
        text = file_handle.read().splitlines()  # 读取后以行进行分割
        file_handle.close()
        return text
    except IOError as error:
        print("Read file Error: {0}".format(error))
        sys.exit()


def compare_file(file1_name, file2_name):
    if file1_name == "" or file2_name == "":
        print('文件路径不能为空：file1_name的路径为：{0}, file2_name的路径为：{1} .'.format(file1_name, file2_name))
        sys.exit()
    text1_lines = read_file(file1_name)
    text2_lines = read_file(file2_name)
    diff = difflib.HtmlDiff()  # 创建htmldiff 对象
    result = diff.make_file(text1_lines, text2_lines)  # 通过make_file 方法输出 html 格式的对比结果
    #  将结果保存到result.html文件中并打开
    try:
        with open('Kg2text/result.html', 'w') as result_file:      #同 f = open('result.html', 'w') 打开或创建一个result.html文件
            result_file.write(result)                      #同 f.write(result)
    except IOError as error:
        print("写入html文件错误：{0}".format(error))

def args_script2lst(file_name):

    with open(file_name, "r") as f:
        lines = f.readlines()
    f.close()

    write_to = file_name+"_args_lst.txt"
    with open(write_to, "a") as f:
    
        args_lst = []
        for line in lines:
            line = line.split()
            if "\\" in line:
                line.remove("\\")
            for i in range(len(line)):
                line[i] = "\""+line[i]+"\""
            args_lst += line
        
            f.write(", ".join(line)+","+"\n")
    
    """
    with open(write_to, "a") as f:
        f.write(", ".join(args_lst))
    return args_lst
    """


if __name__ == "__main__":
    #compare_file("Kg2text/data-bin/webnlg/sentencepiece_bped_data/eval.label", "Kg2text/data-bin/webnlg/sentencepiece_bped_data/check_eval.txt")                   #传入两文件的路径

    string = args_script2lst("/home/xianjiay/efs-storage/fairseq/Kg2text/experiment/denoising_args")

    dict_path = "Kg2text/data-bin/webnlg/data_mbart50_wotags/dict.input.txt"
    path = "Kg2text/data-bin/webnlg/sentencepiece_bped_data/eval.label"
    tgt_dict = Dictionary.load(dict_path)
    data = IndexedRawTextDataset(path, tgt_dict, append_eos=True)

    print("done")



