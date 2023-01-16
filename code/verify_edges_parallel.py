import time 
print("Initiate data and import packages. Current time: " + str(time.strftime("%H:%M:%S", time.localtime())))

from joblib import Parallel, delayed

import pandas as pd
import numpy as np
# import cdd
import sys
import signal
import copy



# Metadata
num_process = 4

parts = 100
start_parts = 0

max_time_per_poly = 10 # Set to 0 if no timeout is wanted

max_dim_cdd = 0

# Read the given data
print("Trying to read the data. This can take some time. ")
imset_list_read = eval(open(sys.argv[2], 'r').read())
print("\tGot a list of "+str(len(imset_list_read))+" imsets")

readfile = sys.argv[1]
savefile = sys.argv[1]

edge_df = pd.read_csv(readfile)
print("\tDone!")

# Timeout handler
class TimeoutException(Exception):
    "Function timed out!"
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout")

signal.signal(signal.SIGALRM, timeout_handler)



# Definte the cdd wrapper to clean the code
def adjacencies_list(mat):
    poly = cdd.Polyhedron(mat)
    # get the adjacent vertices of each vertex
    return [list(x) for x in poly.get_input_adjacency()]

# The numerical verify edge function
def verify_edge_function(alpha, beta, input_vertices, iterations = 100, step = 1e-5, abs_tol = 1e-10):
    dif = alpha-beta
    len_dif = np.inner(dif, dif)
    cost = np.array([0 if dif[k] == 0 else 1+np.random.uniform() for k in range(len(alpha))]) 
    # TODO: Add check that cost is not parallell with dif
    
    # Remove alpha and beta from input_vertices if they are there.
    vertices = input_vertices.copy()
    i = 0
    while i<len(vertices):
        if np.array_equal(vertices[i], alpha) or np.array_equal(vertices[i], beta):
            vertices.pop(i)
            i -= 1
        i += 1
    
    # Project cost to the orthogonal space of dif
    cost = cost - (np.inner(dif, cost)/len_dif) * dif
    value = np.inner(cost, alpha)
    # Improve the 
    for iter in range(iterations):
        check = True
        for v in vertices:
            if np.inner(cost, v) > value:
                # Update cost
                cost_v = np.array([1 if v[k] == 1 else -1 for k in range(len(v))])
                p = cost_v - (np.inner(dif, cost_v)/len_dif)*dif
                scale = -np.inner(cost, v-alpha)/np.inner(p, v-alpha) - np.sign(np.inner(p, v-alpha))*step
                cost = cost+scale*p 
                cost = (1/np.linalg.norm(cost))*cost
                # Make sure the cost is in the orhtogonal space of dif
                # Should not be required since both cost and p already are,
                # but the rescaling seems to propagate errors.
                cost = cost - (np.inner(dif, cost)/len_dif) * dif
                value = np.inner(cost, alpha)
                check = False
        if check:
            vert_copy = []
            for v in vertices:
                if abs(np.inner(cost, v) - value) < abs_tol:
                    vert_copy.append(v)
            vertices = vert_copy.copy()
            # If we have removed all other vertices (which we hopefully have)
            # then we are sure that we have an edge, and can return 1
            if len(vertices) < 2:
                return 1
    # Otherwise no verification was found, and we return 0
    return 0


# Definte a function that checks new information
# returns a list of all rows that we have information about. 
def update_row(row, imset_list, dataframe, max_dim_cdd = max_dim_cdd):
    # If the row is already updated, do nothing.
    if row['edge'] != 0:
        return []
    # Find all vertices that shares all common coordinates
    # or put another way, consider the smallest face of the 
    # cube containing both a_vec and b_vec. 
    temp_list = []
    sum_vec = imset_list[row['a']]+imset_list[row['b']]
    cost = sum_vec - np.ones(len(sum_vec))
    target = sum(k for k in cost if k > 0)
    for j in range(len(imset_list)):
        if np.inner(cost, imset_list[j]) == target:
            temp_list.append(imset_list[j])
    num_vert = len(temp_list)
    # Try to use a quick verification
    if verify_edge_function(imset_list[row['a']], imset_list[row['b']], temp_list, iterations = 100) == 1:
        return [(row.name, 2)] 
    # If the quick verification is not possible, try to use 
    # the cdd algorithm if the dimension is low enough. We do
    # this with a timeout as to not get stuck in lenghty computations
#    if row['dim'] < max_dim_cdd:
#        try:
#            # Run cdd with a timeout (in case computations take too long) 
#            signal.alarm(max_time_per_poly)
#            # TODO: check this next row some day
#            adj_list = adjacencies_list(cdd.Matrix(temp_list))
#            # Cancel the alarm if computations are done
#            signal.alarm(0)
#            # Return all new info
#            ret = []
#            for a in range(num_vert):
#                for b in range(a+1, num_vert):
#                    temp_list_a_vec_id = sum([temp_list[a][k]*3**k for k in range(len(temp_list[a]))])
#                    temp_list_b_vec_id = sum([temp_list[b][k]*3**k for k in range(len(temp_list[b]))]) 
#                    
#                    if b in adj_list[a]:
#                        ret.append((dataframe.loc[(dataframe['a_vec_id'] == temp_list_a_vec_id) & (dataframe['b_vec_id'] == temp_list_b_vec_id)].index[0], 1))
#                    else:
#                        for ind in dataframe.loc[(dataframe['a_vec_id'] == temp_list_a_vec_id) & (dataframe['b_vec_id'] == temp_list_b_vec_id) & (dataframe['edge'] == 0)].index:
#                            ret.append((ind, -1))
#            return ret
#        except TimeoutException:
#            pass
    # If we failed to either verify or not verify 
    # the edge, return 0.
    return [(row.name, 0)]

# Initiate variables
len_df = edge_df.shape[0]
np_imset_list_read = [np.array(i) for i in imset_list_read]

# Start the run. Print starting message and save the time.
print("Starting first run. Current time: " + str(time.strftime("%H:%M:%S", time.localtime())))
st_real = time.time()

edge_df.loc[(edge_df['dim'] < 3) & (edge_df['edge'] == 0), 'edge'] = 1

# This is done in parts and is written to file each time. Notice that
# this append on the previous part. 

for part in range(start_parts, parts):
    print("\nStarting on part: " + str(part+1) + "/" + str(parts))
#    ret = Parallel(n_jobs = num_process, require = 'sharedmem')(delayed(update_row)(row, imset_list_read, edge_df) for ind,row in edge_df[int(part * len_df/parts): int((part+1)*len_df/parts)].loc[edge_df['edge'] == 0].iterrows())
    ret = Parallel(n_jobs = num_process)(delayed(update_row)(row, np_imset_list_read, edge_df) for ind,row in edge_df[int(part * len_df/parts): int((part+1)*len_df/parts)].loc[edge_df['edge'] == 0].iterrows())
    info = list.join(ret)
    for ed in info:
        if edge_df.at[ed[0], 'edge'] == 0:
            edge_df.at[ed[0], 'edge'] = ed[1]
    edge_df.to_csv(savefile, index = False)
    print("Part " + str(part+1) + " completed. Updated "+str(sum(1 for i in info if i[1] != 0))+" rows. Failed to do " +str(sum(1 for i in info if i[1] == 0))+ " rows. Current time: " +str(time.strftime("%H:%M:%S", time.localtime())))
    print("Time taken so far: ", time.time()-st_real)

# ret = edge_df.apply(update_row, args=(edge_df,), axis=1)

print("Program done. Time taken: ", time.time()-st_real)
print("Current time: " + str(time.strftime("%H:%M:%S", time.localtime())))


