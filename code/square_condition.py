import time 
print("Initiate data and import packages. Current time: " + str(time.strftime("%H:%M:%S", time.localtime())))

import sys
import pandas as pd

# Metadata
savefile = sys.argv[1]
readfile = sys.argv[1]


print("Trying to read the data. This can take some time.")
df = pd.read_csv(readfile)
print("Database loaded.")

print("Starting run. Current time: " + str(time.strftime("%H:%M:%S", time.localtime())))
st_real = time.time()
st_cpu = time.process_time()

# Sort the dataframe by vec_sum_id
print("Sorting the dataframe.")

df.sort_values(by = 'vec_sum_id', inplace = True, ignore_index = True)

print("Dataframe sorted. Checking square criterion. Current time: "  + str(time.strftime("%H:%M:%S", time.localtime())))


df.loc[(df['vec_sum_id'] == df['vec_sum_id'].shift(1)) | (df['vec_sum_id'] == df['vec_sum_id'].shift(-1)), 'edge'] = -2
        
print("Square criterion checked. Saving to file.")

df.to_csv(savefile, index= False)

print("Program done. Time taken: ", time.time()-st_real, "Cpu-time taken: ", time.process_time()-st_cpu)

print("Current time: " + str(time.strftime("%H:%M:%S", time.localtime())))




