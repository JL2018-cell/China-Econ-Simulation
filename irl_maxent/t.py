import pdb
i = int(input())
print(i)
if i == 1:
    pdb.set_trace()
else:
    print("Not one.")

j = 0
print("j =", j)
