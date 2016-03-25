local HighwayLayer, parent = torch.class('nn.HighwayLayer', 'nn.Sequential')

function HighwayLayer:__init(sz, transfer, bias, dropout)
	parent.__init(self)
--[=[
	Ultimately, we need transfer(x) * sigmoid(x) + (1-sigmoid(x)) * x
	x
	-> {transfer(x), gate(x), x}                  | components
	-> {gate(x) * transfer(x), (1-gate(x)), x}    | stage1
	-> {gate(x) * transfer(x), x*(1-gate(x))}     | stage2
	-> {gate(x) * transfer(x) + x * (1-gate(x))}  | stage3
]=]
	local components = nn.ConcatTable()
	local gate = nn.Sequential()
	gate:add(nn.Linear(sz, sz))
	gate:add(nn.AddConstant(bias, true))
	gate:add(nn.Sigmoid())
	components:add(gate)

	local carry = nn.Sequential()
	carry:add(nn.Linear(sz, sz))
	carry:add(transfer:clone())
	components:add(carry)


	components:add(nn.Identity())
	self:add(components)

	local stage1 = nn.ConcatTable()
		local stage1gt = nn.Sequential()
		local pieces = nn.ConcatTable()
		pieces:add(nn.SelectTable(1))
		pieces:add(nn.SelectTable(2))
		stage1gt:add(pieces)
		stage1gt:add(nn.CMulTable())
		stage1:add(stage1gt)

		local stage1oneminusg = nn.Sequential()
		stage1oneminusg:add(nn.SelectTable(1))
		stage1oneminusg:add(nn.MulConstant(-1, true))
		stage1oneminusg:add(nn.AddConstant(1, true))
		stage1:add(stage1oneminusg)

		stage1:add(nn.SelectTable(3))

	self:add(stage1)

	local stage2 = nn.ConcatTable()
		stage2:add(nn.SelectTable(1))
		local stage2recall = nn.Sequential()
		local pieces = nn.ConcatTable()
		pieces:add(nn.SelectTable(2))
		pieces:add(nn.SelectTable(3))
		stage2recall:add(pieces)
		stage2recall:add(nn.CMulTable())
		stage2:add(stage2recall)
	
	self:add(stage2)
	self:add(nn.CAddTable())
	self:add(nn.Dropout(dropout, true))

end
