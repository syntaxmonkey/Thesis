import numpy as np
import matplotlib.pyplot as plt
import Bridson_Common

a = []
a.append([0,0])
a.append([1,3])
a.append([2,3.5])
a.append([3,5])
a.append([7,7])

a = np.array(a)

maxLength = Bridson_Common.findLongestSegmentLength( a )
print("maxLength:", maxLength)

plt.plot( a[:,0], a[:,1], linewidth=2 )

plt.show()