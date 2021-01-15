import nn
import numpy as np

loop_num = 300


def main():
    coef_x = []
    coef_t = []
    for i in range(loop_num):
        x, t = nn.main()
        coef_x.append(x)
        coef_t.append(t)
        print(str(i+1)+" / "+str(loop_num))
    print("coef_x ave :", np.array(coef_x).mean())
    print("coef_t ave :", np.array(coef_t).mean())


if __name__ == "__main__":
    main()
