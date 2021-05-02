
from prio_replay import PrioReplay

replay_len = 3
batch_size = 6

mytest = PrioReplay(replay_len,batch_size)

mytest.add([1,1,1,1,False])
mytest.add([1,2,1,1,True])
mytest.add([1,3,1,1,True])

print(mytest.buffer)

mytest.add([1,4,1,1,True])

print(mytest.buffer)

print(mytest.sample())