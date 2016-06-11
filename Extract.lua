local Extract, parent = torch.class('nn.Extract', 'nn.Module')

--[=[
So.
	Our input is a table {indicators, data}
	This module will simply remove any entries in data for which their
	indicator is > 0.5
	See Implant.lua. This module is meant to be used something like this:
		mlp = nn.Sequential()
		slab1 = nn.ConcatTable()
		slab1:add(nn.SelectTable(1))
		slab1:add(nn.Extract())
		mlp:add(slab1)

		slab2 = nn.ParallelTable()
		slab2:add(nn.Identity())

		slab2:add(fancyneural network module here!) -- This is where you put your goodies.

		mlp:add(slab2)

		mlp:add(nn.Implant())

	mlp:forward({torch.ByteTensor(sz):bernoulli(), torch.randn(sz, 2, 3)})
]=]

function Extract:__init()
	parent.__init(self)
	self.gradInput = nil
end

function Extract:updateOutput(input)
	local indicators = input[1]
	local data = input[2]

	local indices = torch.range(1, indicators:numel())[indicators:gt(0.5)]:long()
	self.output = data:index(1,indices)
	return self.output
end

function Extract:updateGradInput(input, gradOutput)
	local indicators = input[1]
	local data = input[2]
	local datasz = data:size()
	local indicatorsz = torch.ones(data:dim())
	indicatorsz[1] = indicators:size(1)
	local expandedindicators = indicators:gt(0.5):resize(torch.LongStorage(indicatorsz:totable())):expand(datasz)

-- Now, we have to put back the rows that we took out.
	self.gradInput = self.gradInput or {
		torch.zeros(indicators:size()):typeAs(indicators),
		torch.Tensor(0):typeAs(data)
	}

	self.gradInput[2]:resizeAs(data):zero()
	self.gradInput[2]:maskedCopy(expandedindicators, gradOutput)
	return self.gradInput
end
