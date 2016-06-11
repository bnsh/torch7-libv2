#! /usr/local/torch/install/bin/th

require "torch"
require "nn"
require "Extract"
require "Implant"

sz = 10
indicators = torch.ByteTensor(sz):bernoulli()
data = torch.randn(sz, 3)
target = torch.randn(sz, 3)

mlp = nn.Sequential()
slab1 = nn.ConcatTable()
slab1:add(nn.SelectTable(1))
slab1:add(nn.Extract())
mlp:add(slab1)

slab2 = nn.ParallelTable()
slab2:add(nn.Identity())
slab2:add(nn.Linear(3,3))

mlp:add(slab2)

mlp:add(nn.Implant())

final = mlp:forward({indicators, data})
criterion = nn.MSECriterion()
err = criterion:forward(final, target)
gradOutput = criterion:backward(final, target)
gradInput = mlp:backward({indicators, data}, gradOutput)

print(final)
print(gradInput[2])
