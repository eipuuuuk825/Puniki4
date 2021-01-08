""" 
data_set_create.py で作成したデータを加工する
"""
import os
import numpy as np
import param as pa
import utility

# パラメータ
path_raw_data_input = "../../data_raw/2021-01-08/"
path_data_set_output = "data/data_set.csv"
first_input_num = 6     # 入力する最初の座標
input_point_num = 12     # 入力データとして使う座標の数
input_point_step = 1    # 入力する座標の間隔
max_point_num = 20      # 入力として使える座標の最大個数
use_y_data = True       # y 座標のデータを使うか
predict_y = 450         # 予測する x, t に対応する y 座標


def calc_teacher_data(x, y, t):
    # yt を超える yi に対応する i を求める
    i = 0
    for yi in y:
        if yi >= predict_y:
            break
        i += 1

    dy1 = y[i] - predict_y
    dy2 = predict_y - y[i-1]
    xt = (dy2*x[i]+dy1*x[i-1])/(dy1+dy2)
    tt = (dy2*t[i]+dy1*t[i-1])/(dy1+dy2)

    return xt, tt


#
# データを正規化する
#
def normalize(x, y, t, xt, tt):
    # エラーチェック
    input_len = [len(x), len(y), len(t), len(xt), len(tt)]
    if(input_len.count(input_len[0]) != len(input_len)):
        print("in normalize, invalid size.")
        print(input_len)
        exit()

    # データを合体する
    x_input = np.array(x)[:, first_input_num:first_input_num + input_point_num *
                          input_point_step:input_point_step]
    y_input = np.array(y)[:, first_input_num:first_input_num + input_point_num *
                          input_point_step:input_point_step]
    t_input = np.array(t)[:, first_input_num:first_input_num + input_point_num *
                          input_point_step:input_point_step]

    data_array = np.array(x_input)
    if use_y_data:
        data_array = np.append(data_array, y_input, axis=1)
    data_array = np.append(data_array, t_input, axis=1)
    xt_array = np.array(xt).reshape(len(xt), 1)
    tt_array = np.array(tt).reshape(len(tt), 1)
    data_array = np.append(data_array, xt_array, axis=1)
    data_array = np.append(data_array, tt_array, axis=1)

    # 平均，標準偏差を求める
    mean = data_array.mean(0)
    std = data_array.std(0)

    # 正規化
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            data_array[i, j] = (data_array[i, j] - mean[j])/std[j]

    return data_array, mean, std


def main():
    #
    # ファイル名一覧を取得
    #
    files = os.listdir(path_raw_data_input)
    files = [f for f in files if os.path.isfile(os.path.join(path_raw_data_input, f))]

    #
    # in_file からデータを読み込み，加工する
    #
    x = []
    y = []
    t = []
    xt = []
    tt = []
    for in_file in range(len(files)):
        xi = []
        yi = []
        ti = []
        with open(path_raw_data_input+files[in_file]) as in_fileobj:
            while True:  # 末尾まで
                line = in_fileobj.readline()
                if line:
                    # １行の文字列をコンマ区切りでリストに
                    data_list = [float(val) for val in line.split(
                        ",") if utility.is_num(val)]
                    xi.append(data_list[0])
                    yi.append(data_list[1])
                    ti.append(data_list[2])
                else:
                    break
        x.append(xi[0:max_point_num])
        y.append(yi[0:max_point_num])
        t.append(ti[0:max_point_num])

        # 教師データ xt, tt を求める
        xti, tti = calc_teacher_data(xi, yi, ti)
        xt.append(xti)
        tt.append(tti)

    # データを規格化
    normalised_data, mean, std = normalize(x, y, t, xt, tt)

    #
    # データセットを書き出す
    #
    sep = ","  # 区切る文字
    with open(path_data_set_output, "w") as fileobj:
        # yt
        fileobj.write("yt\n"+str(predict_y)+"\n")
        # 平均
        fileobj.write("mean\n")
        for i in mean:
            fileobj.write(str(i)+sep)
        fileobj.write("\n")
        # 標準偏差
        fileobj.write("std\n")
        for i in std:
            fileobj.write(str(i)+sep)
        fileobj.write("\n")
        # x, y, t, xt, tt
        fileobj.write("data_set\n")
        for i in normalised_data:
            for j in i:
                fileobj.write(str(j)+sep)
            fileobj.write("\n")


if __name__ == '__main__':
    main()
