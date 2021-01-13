def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


E = []

with open("E.csv") as fileobj:
    while True:
        line = fileobj.readline()
        if line:
            num = line.split("\n")[0]
            if is_num(num):
                E.append(float(num))
        else:
            break

pre_val = 1E9
err_cnt = 0
for i in range(len(E)):
    if pre_val < E[i]:
        err_cnt += 1
        # print("error ", i, "/", len(E), pre_val, E[i])
    pre_val = E[i]

print(err_cnt)
