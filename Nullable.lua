local Nullable, parent = torch.class('nn.Nullable', 'nn.Sequential')

--[=[
So.
	Our input is a table {indicators, data}
	See Extract.lua and Implant.lua in this same directory.
]=]

function Nullable:__init(realmodule)
	parent.__init(self)
	local slab1 = nn.ConcatTable()
	slab1:add(nn.SelectTable(1))
	slab1:add(nn.Extract())
	self:add(slab1)

	slab2 = nn.ParallelTable()
	slab2:add(nn.Identity())
	slab2:add(realmodule)

	self:add(slab2)

	self:add(nn.Implant())
end
