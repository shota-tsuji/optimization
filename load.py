import pandas as pd
import re
import os
import sys

test_file = sys.argv[1]
filename, _ = os.path.splitext(test_file)
tmpname = "tmp-" + filename
outname = filename + ".csv"

ifn = open(test_file, "r")
ofn = open(tmpname, "w")

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

df = pd.read_csv(tmpname, header=None)
n_rows_origin = len(df)

df = df.dropna()
df = df.astype('int64')
n_rows_dropped = len(df)
print("{}/{}".format(n_rows_dropped, n_rows_origin))

df.to_csv(outname, header=False, index=False)
if os.path.exists(tmpname):
    os.remove(tmpname)

