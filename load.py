import pandas as pd
import re
import math

ifn = open("a1a", "r")
ofn = open("tmp.a1a.csv", "w")

while True: 
    line = ifn.readline()
    if not line:
        break
    line = line.replace(":1", "")
    line = line.replace(" $", "")
    line = re.sub(" $", "", line)
    line = line.replace(" ", ",")
    ofn.write(line)

ifn.close()
ofn.close()


df = pd.read_csv('tmp.a1a.csv', header=None)
n_rows_origin = len(df)

df = df.dropna()
n_rows_dropped = len(df)

df = df.astype('int64')
print(df)
df.to_csv('a1a.csv', header=False, index=False)

print("{}/{}".format(n_rows_dropped, n_rows_origin))
