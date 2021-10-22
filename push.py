f = open("data.txt", "w")
f1 = open("requirements.txt", "r")
for i in f1.read().split("\n"):
    f.write("==".join(i.split("==")[:2]))
    f.write("\n")
