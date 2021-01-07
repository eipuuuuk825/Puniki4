# NN ライブラリを使いやすいように変更
# 7763c160b4ec1caa99718cd3c865339227a1908e

import numpy as np
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
    nn.learning(data_training_x, data_training_t, "E.csv", "param.csv")
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
    print(coef_x, coef_t)

    # ファイルに出力
    with open("result.csv", "w") as fileobj:
        fileobj.write("true x, predict x, true t, predict t\n")
        for i in result:
            output_str = str(i[0])+","
            output_str += str(i[1])+","
            output_str += str(i[2])+","
            output_str += str(i[3])+"\n"
            fileobj.write(output_str)


if __name__ == '__main__':
    main()
