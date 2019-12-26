from mpi4py import MPI
import numpy as np
import time
from operator import itemgetter

def minimumOrMaximum(list, minormax):
    if minormax == "max":
        return max(list)
    return min(list)


#Assign global variables
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    arraySize = input("Please enter array size: ")

    #Generate numbers of size n
    numbers = np.random.choice(999999, int(arraySize))
    # print("Generated list of size " + str(arraySize) + " is: " + str(numbers))
    minormax = input("Please enter \"min\" or \"max\" choice: ")

    chunks = np.array_split(numbers, size)
else:
    chunks = None
    minormax = None

#start script with parallel processes
start_time = time.time()

#send minormax value to all workers
minormax = comm.bcast(minormax, root=0)
comm.barrier()

chunk = comm.scatter(chunks, root=0)
# print("Process " + str(rank) +" has this chunk of data: " + str(chunk))

chunk = minimumOrMaximum(chunk, minormax)
print("Process " + str(rank) +" has this minimum value: " + str(chunk)) 

sortedArrays = comm.gather(chunk, root=0)

if rank == 0:
    minimumValue = minimumOrMaximum(sortedArrays, minormax)
    
    print("\n\n Minimum Value: " + str(minimumValue))

    #End of script
    print("\n\n Execution Time --- %s seconds ---" % (time.time() - start_time))
