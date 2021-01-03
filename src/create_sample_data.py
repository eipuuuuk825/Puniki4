""" 
main.py で取得したデータからサンプルデータを作成する
"""
import os
import datetime  # 現在時刻取得
import numpy as np

# パラメータ
data_raw_dir = "../data_raw/"
data_sample_dir = "../data_sample/"
yt = 350            # x と t を予測する y 座標
input_point_num = 6  # 入力データとして使う座標の数


def calc_teacher_data(x, y, t):
    # yt を超える yi に対応する i を求める
    i = 0
    for yi in y:
        if yi >= yt:
            break
        i += 1

    dy1 = y[i] - yt
    dy2 = yt - y[i-1]
    xt = (dy2*x[i]+dy1*x[i-1])/(dy1+dy2)
    tt = (dy2*t[i]+dy1*t[i-1])/(dy1+dy2)

    return xt, tt


#
# サンプルデータを正規化する
#
def normalize(x, y, t, xt, tt):
    # エラーチェック
    input_len = [len(x), len(y), len(t), len(xt), len(tt)]
    if(input_len.count(input_len[0]) != len(input_len)):
        print("in normalize, invalid size.")
        print(input_len)
        exit()

    # データを合体する
    data_array = np.array(x)
    data_array = np.append(data_array, y, axis=1)
    data_array = np.append(data_array, t, axis=1)
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

    return data_array


def main():
    #
    # ファイル名一覧を取得
    #
    files = os.listdir(data_raw_dir)
    files = [f for f in files if os.path.isfile(os.path.join(data_raw_dir, f))]

    #
    # in_file からサンプルデータを読み込み，加工する
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
        with open(data_raw_dir+files[in_file]) as in_fileobj:
            while True:  # 末尾まで
                line = in_fileobj.readline()
                line = line.rstrip()   # 末尾の改行を削除
                if line:
                    # １行の文字列をコンマ区切りでリストに
                    data_list = [float(val) for val in line.split(", ")]
                    xi.append(data_list[0])
                    yi.append(data_list[1])
                    ti.append(data_list[2])
                else:
                    break
        x.append(xi[0:input_point_num])
        y.append(yi[0:input_point_num])
        t.append(ti[0:input_point_num])

        # 教師データ xt, tt を求める
        xti, tti = calc_teacher_data(xi, yi, ti)
        xt.append(xti)
        tt.append(tti)

    normalised_data = normalize(x, y, t, xt, tt)

    #
    # サンプルデータを書き出す
    #
    out_file = data_sample_dir + \
        str(datetime.datetime.now()).replace(":", ".")+".csv"
    with open(out_file, "w") as out_fileobj:
        for i in normalised_data:
            for j in i:
                out_fileobj.write(str(j)+" ")
            out_fileobj.write("\n")

if __name__ == '__main__':
    main()
