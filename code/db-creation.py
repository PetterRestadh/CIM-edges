import time 
print("Initiate data and import packages. Current time: " + str(time.strftime("%H:%M:%S", time.localtime())))

from joblib import Parallel, delayed

import pandas as pd
import numpy as np
import sys
import gc


# Metadata
num_process = 4
parts = 10
start_parts = 0

imset_list = eval(open(sys.argv[1], 'r').read())
imset_list_id = [sum([imset_list[i][k]*(3**k) for k in range(len(imset_list[i]))]) for i in range(len(imset_list))]

# Optional options (run only for a small part of the data)

if len(sys.argv) > 3:
    num_vert = min(int(sys.argv[3]), len(imset_list))
else:
    num_vert = len(imset_list)
    
if start_parts == 0:
    pd.DataFrame(columns = ['a', 'a_vec_id', 'b', 'b_vec_id', 'edge', 'vec_sum_id', 'dim']).to_csv(sys.argv[2], index = False)

# Start the run. Print starting message and save the time.
print("Starting first run. Current time: " + str(time.strftime("%H:%M:%S", time.localtime())))
st_cpu = time.process_time()
st_real = time.time()

# Definte the important function
def partial_df(i, input_imsets, imset_list_id, max_b = None):
    if max_b == None:
        num_vert = len(imset_list)
    else:
        num_vert = max_b
    vec_i = np.array(input_imsets[i])
    temp_df = pd.DataFrame(columns = ['a', 'a_vec_id', 'b', 'b_vec_id', 'edge', 'vec_sum_id', 'dim'], index = range(i+1, num_vert))
    for j in range(i+1, num_vert):
        temp_df.at[j, 'a'] = i
#        temp_df.at[j, 'a_vec'] = np.array(imset_list[i])
        temp_df.at[j, 'a_vec_id'] = imset_list_id[i]
        temp_df.at[j, 'b'] = j
#        temp_df.at[j, 'b_vec'] = np.array(imset_list[j])
        temp_df.at[j, 'b_vec_id'] = imset_list_id[j]
        temp_df.at[j, 'edge'] = 0
        temp_df.at[j, 'vec_sum_id'] = imset_list_id[i] + imset_list_id[j]
        temp_df.at[j, 'dim'] = sum([1 for k in range(len(imset_list[i])) if imset_list[i][k] != imset_list[j][k]])
    return temp_df

# Create the data frame

# This is done in parts and is written to file each time. Notice that
# this append on the previous part. 

for part in range(start_parts, parts):
    print("Starting on part: ", part+1, "/", parts)
    df_list = Parallel(n_jobs = num_process)(delayed(partial_df)(i, imset_list, imset_list_id, max_b = num_vert) for i in range(int(part * num_vert/parts), int((part+1)*num_vert/parts)))
    for l in df_list:
        l.to_csv(sys.argv[2], index = False, header = False, mode = 'a')
    del df_list
    gc.collect()
    print("Part ", part+1, " completed. Current time: ", time.strftime("%H:%M:%S", time.localtime()))
    print("Time taken so far: ", time.time()-st_real)


print("Program done. Time taken: ", time.time()-st_real, "Cpu-time taken: ", time.process_time()-st_cpu)

print("Current time: " + str(time.strftime("%H:%M:%S", time.localtime())))


