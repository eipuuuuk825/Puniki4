# NN ライブラリを使いやすいように変更
# 7763c160b4ec1caa99718cd3c865339227a1908e

import numpy as np
import matplotlib.pyplot as plt
import so

# 入力された文字列が数値を表すかを判定


def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def main():
    #
    # ファイルからデータセットを読み込む
    #
    with open("../../data_set/1.csv") as fileobj:
        # 平均，標準偏差
        line_list = [float(val) for val in fileobj.readline().split(",")]
        xt_mean = line_list[0]
        xt_std = line_list[1]
        tt_mean = line_list[2]
        tt_std = line_list[3]

        # x, y, t
        data_set = []
        while True:  # 末尾まで
            line = fileobj.readline()
            if line:
                line_list = [float(val)
                             for val in line.split(",") if is_num(val)]
                data_set.append(line_list)
            else:
                break

    #
    # NN に渡すデータを用意
    #
    data_training_x = np.array(data_set)[:50, :-2]
    data_training_t = np.array(data_set)[:50, -2:]
    data_test_x = np.array(data_set)[50:, :-2]
    data_test_t = np.array(data_set)[50:, -2:]

    #
    # NN
    #
    nn = so.NeuralNetwork([18, 10, 2], 0.1, "regression")
    nn.learning(data_training_x, data_training_t,
                "data/E.csv", "data/param.csv")
    # nn = so.NeuralNetwork("regression", "param.csv")

    # 精度を検証
    predict_x = []
    predict_t = []
    for i in range(data_test_x.shape[0]):
        predict = nn.compute(data_test_x[i])
        predict_x.append(predict[0])
        predict_t.append(predict[1])
    result = data_test_t[:, :1]
    result = np.append(result, np.array(
        predict_x).reshape(len(predict_x), 1), axis=1)
    result = np.append(result, data_test_t[:, 1:], axis=1)
    result = np.append(result, np.array(
        predict_t).reshape(len(predict_t), 1), axis=1)

    # 相関係数
    coef_x = np.corrcoef(result[:, 0:2].T)[0, 1]
    coef_t = np.corrcoef(result[:, 2:].T)[0, 1]
    # print(coef_x, coef_t)

    # ファイルに出力
    with open("data/result.csv", "w") as fileobj:
        fileobj.write("true x, predict x, true t, predict t,,")
        fileobj.write("coef x,"+str(coef_x)+",coef t,"+str(coef_t)+"\n")
        for i in result:
            output_str = str(i[0])+","
            output_str += str(i[1])+","
            output_str += str(i[2])+","
            output_str += str(i[3])+"\n"
            fileobj.write(output_str)

    #
    # 散布図を描画
    #
    fig = plt.figure(figsize=(12, 6))
    # x
    xy_max = np.max(result[:, 0:2])
    xy_min = np.min(result[:, 0:2])
    g_x = fig.add_subplot(1, 2, 1)
    plt.scatter(result[:, 0:1], result[:, 1:2])
    plt.plot([xy_min, xy_max], [xy_min, xy_max], color="orange")
    plt.title("predict x ("+f'{coef_x:.3f}'+")")
    plt.xlabel("true x [px]")
    plt.ylabel("predict x [px]")
    plt.grid(True)
    # t
    xy_max = np.max(result[:, 2:4])
    xy_min = np.min(result[:, 2:4])
    g_t = fig.add_subplot(1, 2, 2)
    plt.scatter(result[:, 2:3], result[:, 3:4])
    plt.plot([xy_min, xy_max], [xy_min, xy_max], color="orange")
    plt.title("predict t ("+f'{coef_t:.3f}'+")")
    plt.xlabel("true t [sec]")
    plt.ylabel("predict t [sec]")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
