# NN ライブラリを使いやすいように変更
# 7763c160b4ec1caa99718cd3c865339227a1908e

import numpy as np
import matplotlib.pyplot as plt


def main():
    # ファイルからプロットする値を読み込む
    true_x = []
    predict_x = []
    with open("data/tmp_result.csv") as fileobj:
        while True:
            line = fileobj.readline()
            if line:
                line = line.rstrip()
                splitted = line.split(",")
                true_x.append(float(splitted[0]))
                predict_x.append(float(splitted[1]))
            else:
                break

    # 相関係数
    coef = np.corrcoef(true_x, predict_x)[0, 1]
    print(coef)

    # 散布図を描画
    font_size = 15
    fig = plt.figure(figsize=(12, 6))
    # x
    xy_max = np.max(true_x+predict_x)
    xy_min = np.min(true_x+predict_x)
    g_x = fig.add_subplot(1, 2, 1)
    plt.plot([xy_min, xy_max], [xy_min, xy_max], color="orange")
    plt.scatter(true_x, predict_x)
    plt.title("xt (相関係数："+f'{coef:.3f}'+")",
                fontname="MS Gothic", fontsize=font_size)
    plt.xlabel("xt の真値 [px]", fontname="MS Gothic", fontsize=font_size)
    plt.ylabel("xt の予測値 [px]", fontname="MS Gothic", fontsize=font_size)
    plt.grid(True)
    # t
    # xy_max = np.max(result[:, 2:4])
    # xy_min = np.min(result[:, 2:4])
    # g_t = fig.add_subplot(1, 2, 2)
    # plt.plot([xy_min, xy_max], [xy_min, xy_max], color="orange")
    # plt.scatter(result[:, 2:3], result[:, 3:4])
    # plt.title("tt (相関係数："+f'{coef_t:.3f}'+")",
    #             fontname="MS Gothic", fontsize=font_size)
    # plt.xlabel("tt の真値 [sec]", fontname="MS Gothic", fontsize=font_size)
    # plt.ylabel("tt の予測値 [sec]", fontname="MS Gothic", fontsize=font_size)
    # plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
