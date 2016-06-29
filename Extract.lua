local Extract, parent = torch.class('nn.Extract', 'nn.Module')

--[=[
So.
	Our input is a table {indicators, data}
	This module will simply remove any entries in data for which their
	indicator is > 0.5
	See Implant.lua, Nullable.Lua. This module is meant to be used something like this:
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

	(or more conveniently: mlp = Nullable(fancyneural network module here!))

	mlp:forward({torch.ByteTensor(sz):bernoulli(), torch.randn(sz, 2, 3)})
]=]

function Extract:__init()
	parent.__init(self)
	self.gradInput = nil
	self.output = nil
end

function Extract:updateOutput(input)
	local indicators = input[1]
	local data = input[2]

	self.output = self.output or torch.Tensor(0):typeAs(data)

	local rv = data:index(1, torch.LongTensor({1}))
	if indicators:gt(0.5):any() then
		local indices = torch.range(1, indicators:numel())[indicators:gt(0.5):byte()]:long()
		rv = data:index(1,indices)
	end
	self.output:resizeAs(rv)
	self.output:copy(rv)
	return self.output
end

function Extract:updateGradInput(input, gradOutput)
	local indicators = input[1]
	local data = input[2]

-- Now, we have to put back the rows that we took out.
	self.gradInput = self.gradInput or {
		torch.ByteTensor(0),
		torch.Tensor(0):typeAs(data)
	}

	self.gradInput[1]:resizeAs(indicators):zero()
	self.gradInput[2]:resizeAs(data):zero()

	if indicators:gt(0.5):any() then
		local indices = torch.range(1,indicators:numel())[indicators:gt(0.5):byte()]:long()
		self.gradInput[2]:indexCopy(1, indices, gradOutput)
	end
	return self.gradInput
end
