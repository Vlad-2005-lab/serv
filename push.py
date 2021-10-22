summ = 0
for i in range(1, 998):
    summ += int(str(i)[0]) - int(str(i)[-1])
print(summ)
