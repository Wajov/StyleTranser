from matplotlib import pyplot


file = open('output/loss_G.txt', 'r')
loss_G = []
for line in file:
    loss_G.append(float(line))
file.close()

file = open('output/loss_D_X.txt', 'r')
loss_D_X = []
for line in file:
    loss_D_X.append(float(line))
file.close()

file = open('output/loss_D_Y.txt', 'r')
loss_D_Y = []
for line in file:
    loss_D_Y.append(float(line))
file.close()

pyplot.subplot(311)
pyplot.xlabel('batch')
pyplot.ylabel('loss_G')
pyplot.plot(loss_G)
pyplot.subplot(312)
pyplot.xlabel('batch')
pyplot.ylabel('loss_D_X')
pyplot.plot(loss_D_X)
pyplot.subplot(313)
pyplot.xlabel('batch')
pyplot.ylabel('loss_D_Y')
pyplot.plot(loss_D_Y)
pyplot.show()
