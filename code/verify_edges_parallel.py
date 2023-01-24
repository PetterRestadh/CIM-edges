import time 
print("Initiate data and import packages. Current time: " + str(time.strftime("%H:%M:%S", time.localtime())))

from joblib import Parallel, delayed

import pandas as pd
import numpy as np
import sys

from scipy.optimize import linprog


# Metadata

parts = 10 # Divide the data into parts and do them seperately
start_parts = 0 # Should generally be 0 unless you return to previous computations.

# Read the given data
print("Trying to read the data. This can take some time. ")
imset_list_read = eval(open(sys.argv[2], 'r').read())
print("\tGot a list of "+str(len(imset_list_read))+" imsets")

readfile = sys.argv[1]
savefile = sys.argv[1]

edge_df = pd.read_csv(readfile)
print("\tDone!")

# The numerical verify edge function
def verify_edge_function(alpha, beta, input_vertices, iterations = 100, step = 1e-5, abs_tol = 1e-10):
    dif = alpha-beta
    len_dif = np.inner(dif, dif)
    cost = np.array([0 if dif[k] == 0 else 1+np.random.uniform() for k in range(len(alpha))]) 
    # TODO: Add check that cost is not parallell with dif
    # Remove alpha and beta from input_vertices if they are there.
    vertices = input_vertices[:]
    i = 0
    while i<len(vertices):
        if np.array_equal(vertices[i], alpha) or np.array_equal(vertices[i], beta):
            vertices.pop(i)
            i -= 1
        i += 1
    
    # Project cost to the orthogonal space of dif
    cost = cost - (np.inner(dif, cost)/len_dif) * dif
    value = np.inner(cost, alpha)
    # Improve the cost function.
    for it in range(iterations):
        check = True
        for v in vertices:
            if np.inner(cost, v) > value:
                # Update cost
                cost_v = np.array([1 if v[k] == 1 else -1 for k in range(len(v))])+ 1e-3*np.random.uniform(size = len(v))
                p = cost_v - (np.inner(dif, cost_v)/len_dif)*dif
                with np.errstate(divide='raise'):
                    try:
                        scale = -np.inner(cost, v-alpha)/np.inner(p, v-alpha) - np.sign(np.inner(p, v-alpha))*step
                        cost = cost+scale*p 
                        cost = (1/np.linalg.norm(cost))*cost
                        # Make sure the cost is in the orhtogonal space of dif
                        # Should not be required since both cost and p already are,
                        # but the rescaling seems to propagate errors.
                        cost = cost - (np.inner(dif, cost)/len_dif) * dif
                        value = np.inner(cost, alpha)
                    except RuntimeWarning:
                        cost = np.array([0 if dif[k] == 0 else 1+np.random.uniform(max = 2) for k in range(len(alpha))])
                        cost = cost-(np.inner(dif, cost)/len_dif) * dif
                        value = np.inner(cost, alpha)
                check = False
        if check:
            vert_copy = []
            for v in vertices:
                if abs(np.inner(cost, v) - value) < abs_tol:
                    vert_copy.append(v)
            vertices = vert_copy[:]
            # If we have removed all other vertices (which we hopefully have)
            # then we are sure that we have an edge, and can return 1
            if len(vertices) < 2:
                if abs(np.inner(cost, dif))< abs_tol: # Final "feel good check"
                    return 1
    # Otherwise no verification was found, and we return 0
    return 0


# linprog version, checks if the projection of alpha
# is in the convex hull of the projection of all vertices.
# It utilizes that the vertices are already given in
# homogenous form (leading 1).
def verify_edge_linprog(alpha, beta, input_vertices):
    dif = alpha-beta
    len_dif = np.inner(dif, dif)
    vertices = [v-(np.inner(v, dif)/len_dif)*dif  for v in input_vertices]
    b = alpha-(np.inner(alpha, dif)/len_dif)*dif
    lp = linprog(np.zeros(len(vertices)), A_eq = np.array(vertices).T, b_eq = b)
    return not lp.success



# Definte a function that checks new information
# returns a list of all rows that we have information about. 
def update_row(row, imset_list, methods = ['numerical', 'linprog']):
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
    if ('numerical' in methods) and verify_edge_function(imset_list[row['a']], imset_list[row['b']], temp_list, iterations = 100) == 1:
        return [(row.name, 2)]
    # Else we rely on scipy's linprog
    if ('linprog' in methods):
        if verify_edge_linprog(imset_list[row['a']], imset_list[row['b']], temp_list):
            return [(row.name, 3)]
        else:
            return [(row.name, -3)]
    # If we failed to either verify or not verify
    # the edge, return 0.
    return [(row.name, 0)]

# Initiate variables
len_df = edge_df.shape[0]
np_imset_list_read = [np.array(i) for i in imset_list_read]
no_updated = 0
no_failed = 0

# Start the run. Print starting message and save the time.
print("Starting first run. Current time: " + str(time.strftime("%H:%M:%S", time.localtime())))
st_real = time.time()

# Take care of some trivial cases first. 
# ONLY WORKS IF SQUARE CRITERION IS TAKEN CARE OF!
# Else this can verify some non-edges.
edge_df.loc[(edge_df['dim'] < 3) & (edge_df['edge'] == 0), 'edge'] = 1

for part in range(start_parts, parts):
    print("\nStarting on part: " + str(part+1) + "/" + str(parts))
    # Collect all returns in ret. 
    ret = []
    for ind,row in edge_df[int(part * len_df/parts): int((part+1)*len_df/parts)].loc[edge_df['edge'] == 0].iterrows():
        ret += update_row(row, np_imset_list_read)
    # Write the new data to file.
    for ed in ret:
        if edge_df.at[ed[0], 'edge'] == 0:
            edge_df.at[ed[0], 'edge'] = ed[1]
    edge_df.to_csv(savefile, index = False)
    # Print some information.
    no_updated += sum(1 for i in ret if i[1] != 0)
    no_failed += sum(1 for i in ret if i[1] == 0)
    print("Part " + str(part+1) + " completed. Updated "+str(sum(1 for i in ret if i[1] != 0))+" rows. Failed to do " +str(sum(1 for i in ret if i[1] == 0))+ " rows. Current time: " +str(time.strftime("%H:%M:%S", time.localtime())))
    print("Time taken so far: "+ str(time.time()-st_real))


# Print end message.
print("Program done. Time taken: " + str(time.time()-st_real))
print("Updated " + str(no_updated) + " entries, failed to do " + str(no_failed)+ " entries.")
print("Current time: " + str(time.strftime("%H:%M:%S", time.localtime())))

