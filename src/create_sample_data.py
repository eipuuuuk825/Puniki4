""" 
main.py で取得したデータからサンプルデータを作成する
"""
import os
import datetime  # 現在時刻取得

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


def main():
    # ファイル名一覧を取得
    files = os.listdir(data_raw_dir)
    files = [f for f in files if os.path.isfile(os.path.join(data_raw_dir, f))]

    # 出力ファイル
    out_file = data_sample_dir+str(datetime.datetime.now()).replace(":", ".")+".csv"

    for in_file in range(len(files)):
        xi = []
        yi = []
        ti = []
        # in_file から xi, yi, ti を読み込む
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

        # 教師データ xt, tt を求める
        xt, tt = calc_teacher_data(xi, yi, ti)

        # 一行ずつサンプルデータを書き出す
        with open(out_file, "a") as out_fileobj:
            # input_point_num 個分の座標データ
            for i in range(input_point_num):
                out_fileobj.write(str(xi[i])+", "+str(yi[i])+", "+str(ti[i])+", ")
            # 教師データ xt, tt
            out_fileobj.write(str(xt)+", "+str(tt)+"\n")


if __name__ == '__main__':
    main()
