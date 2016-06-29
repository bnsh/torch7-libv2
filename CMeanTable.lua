local CMeanTable, parent = torch.class('nn.CMeanTable', 'nn.Module')

--[=[
This is based on CAddTable
We want something that will take the mean of the tables
where the tensor is not zero.
]=]

local function findrowswithdata(tensor)
-- It'd be nice if we could just do something like tensor:sum(2:) to sum up
-- everything _after_ dimension 2, but this seems to not be possible. *sigh*
	local m = tensor:size()[1]
	local n = torch.LongTensor(tensor:size():totable()):prod() / m
	local cpy = tensor:reshape(m,n)
	local nonzero = cpy:sum(2):ne(0):byte() -- We don't want this to be a CudaTensor
	return nonzero
end

local function computeDenominators(data)
	local denominators = nil
	table.foreachi(data, function (key, value)
		if denominators == nil then
			denominators = findrowswithdata(value)
		else
			denominators:add(findrowswithdata(value))
		end
	end)
	return denominators
end

function CMeanTable:__init(ip)
	parent.__init(self)
	self.gradInput = {}
end

function CMeanTable:updateOutput(input)
	self.output:resizeAs(input[1]):copy(input[1])

	for i=2,#input do
		self.output:add(input[i])
	end

	local denominators = computeDenominators(input)
	if denominators:eq(0):any() then
-- Actually, since the only way something will have a denominator of zero,
-- is that every entry in there is zero, it should be safe to set denominator
-- to be one where denominator = 0
		denominators[denominators:eq(0)] = 1
	end
	local sz = torch.LongStorage(torch.ones(self.output:dim()):totable())
	sz[1] = denominators:size(1)
	denominators = denominators:reshape(sz)

	self.output:cdiv(denominators:expandAs(self.output):typeAs(self.output))

	return self.output
end

function CMeanTable:updateGradInput(input, gradOutput)
	local denominators = computeDenominators(input)
	for i=1,#input do
		self.gradInput[i] = self.gradInput[i] or input[1].new()
		self.gradInput[i]:resizeAs(input[i]):copy(gradOutput)

-- Here, where denominator is zero, we need to set self.gradInput to be 0
-- But, elsewhere, we need to divide by the denominator
		local sz = torch.LongStorage(torch.ones(self.gradInput[i]:dim()):totable())
		sz[1] = denominators:size(1)
		denominators = denominators:reshape(sz)

		self.gradInput[i]:cdiv(denominators:expandAs(self.gradInput[i]):typeAs(self.gradInput[i]))

-- Now, we have to zero out any zero vectors in our input.
		local zeros = findrowswithdata(input[i]):eq(0):byte()
		if zeros:any() then
			local indices = torch.range(1, zeros:size(1)):long()[zeros]
			self.gradInput[i]:indexFill(1, indices, 0)
		end
		assert(self.gradInput[i]:ne(self.gradInput[i]):sum() == 0)
	end

	for i=#input+1, #self.gradInput do
		self.gradInput[i] = nil
	end

	return self.gradInput
end
