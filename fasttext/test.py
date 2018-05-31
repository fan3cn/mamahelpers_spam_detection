import os
import numpy as np

test_lines = []
path = "data/"
X_online_test = os.path.join(path, 'X.test')

y_validate = [1,2,3,4,5,6,7,8]

with open(X_online_test) as r_f:
    with open("result.txt", "w") as w_f:
        test_lines = r_f.readlines()
        for line in test_lines:
            w_f.write("asada" + "\t" + line)

# Generate result
#    result = [ [ str(y_validate[i]) + "\t" + X_validate[i] ] for i in range(len(y_validate))]
#result = [ [  str(y_validate[i]) + "\t" + test_lines[i].encode("utf-8") ] for i in range(len(y_validate))]

#print(result)
#with open('result.txt', 'w') as f:
#    f.write(result)
#np.savetxt( "result.txt", result, fmt='%s')
