import matplotlib.pyplot as plt
import sys
import math
epoch = 0
step = 0.0173333
z = []
y = []
count = 0
for line in sys.stdin:
	epoch += step
	perp = [float(x.split('=')[1]) for x in line.split() if x.split('=')[0] == 'perplexity'][0]
	#if perp > 20:
		#perp = 20
	#if count%5 == 0:
	z.append(epoch)
	y.append(math.log(perp))
	count += 1
plt.plot(z,y)
plt.show()
