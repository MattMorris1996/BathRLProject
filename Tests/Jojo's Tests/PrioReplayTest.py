
from PrioReplay import PrioReplay

replay_len = 3

mytest = PrioReplay(replay_len)

mytest.add([1,1],5)
mytest.add([1,2],1)
mytest.add([1,3],2)

print(mytest.sample())

print(mytest.importance)

mytest.add([1,4],4)

print(mytest.sample())

print(mytest.importance)

