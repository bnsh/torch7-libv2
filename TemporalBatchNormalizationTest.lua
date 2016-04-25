#! /usr/local/torch/install/bin/th

require "nn"
require "TemporalBatchNormalization"

mm = nn.Sequential()
mm:add(nn.LookupTable(300,5))
mm:add(nn.TemporalConvolution(5,12,3,1))
mm:add(nn.TemporalBatchNormalization(12))

data = (torch.rand(5,10) * 300):int():add(1)
rv = mm:forward(data)

print(rv:size())
