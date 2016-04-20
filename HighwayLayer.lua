local HighwayLayer, parent = torch.class('nn.HighwayLayer', 'nn.Sequential')

local function OneMinus()
	local mlp = nn.Sequential()
	mlp:add(nn.MulConstant(-1))
	mlp:add(nn.AddConstant(1))
	return mlp
end

local function Gate(sz, bias)
	local mlp = nn.Sequential()
	mlp:add(nn.Linear(sz, sz))
	mlp:add(nn.AddConstant(bias))
	mlp:add(nn.Sigmoid())

	local ct = nn.ConcatTable()
	ct:add(nn.Identity())
	ct:add(OneMinus())
	mlp:add(ct)
	return mlp
end

local function Transfer(sz, transfer)
	local mlp = nn.Sequential()
	mlp:add(nn.Linear(sz, sz))
	mlp:add(transfer:clone())
	return mlp
end

local function GateAndTransfer(sz, transfer, bias)
	local mlp = nn.ConcatTable()
	mlp:add(Gate(sz, bias))
	mlp:add(Transfer(sz, transfer))
	return mlp
end

local function DroppedOutGateAndTransfer(sz, transfer, bias, dropout)
	local mlp = nn.Sequential()
	mlp:add(nn.Dropout(dropout)) -- DON'T try to inline this!
	mlp:add(GateAndTransfer(sz, transfer, bias))
	return mlp
end

function HighwayLayer:__init(num_layers, sz, transfer, bias, dropout)
	parent.__init(self)
--[=[
	Ultimately, we need transfer(dropout(x)) * sigmoid(dropout(x)) + (1-sigmoid(dropout(x))) * x
	It _needs_ (maybe) to be the same dropout on all of these!
]=]
	dropout = dropout or 0.0 -- No dropout if it's not specified.
	for i = 1,num_layers do
		local stage1 = nn.ConcatTable()
			stage1:add(nn.Identity())
			stage1:add(DroppedOutGateAndTransfer(sz, transfer, bias, dropout))
		self:add(stage1)
		self:add(nn.FlattenTable()) -- Presumably, now we have {X, S(D(X)), (1-S(D(X))), T(D(X))} in a table.

		local stage2 = nn.ConcatTable()
			local stage2carry = nn.Sequential()
				local carrygather = nn.ConcatTable()
				carrygather:add(nn.SelectTable(1))
				carrygather:add(nn.SelectTable(3))
				stage2carry:add(carrygather)
				stage2carry:add(nn.CMulTable())
			stage2:add(stage2carry)
			local stage2transfer = nn.Sequential()
				local transfergather = nn.ConcatTable()
				transfergather:add(nn.SelectTable(2))
				transfergather:add(nn.SelectTable(4))
				stage2transfer:add(transfergather)
				stage2transfer:add(nn.CMulTable())
			stage2:add(stage2transfer)
		self:add(stage2)
		self:add(nn.CAddTable())
	end

end
--[=[
nn.Sequential {
  [input -> (1) -> output]
  (1): nn.Sequential {
    [input -> (1) -> (2) -> (3) -> (4) -> output]
    (1): nn.ConcatTable {
      input
        |`-> (1): nn.Identity
        |`-> (2): nn.Sequential {
        |      [input -> (1) -> (2) -> output]
        |      (1): nn.Dropout(0.000000)
        |      (2): nn.ConcatTable {
        |        input
        |          |`-> (1): nn.Sequential {
        |          |      [input -> (1) -> (2) -> (3) -> (4) -> output]
        |          |      (1): nn.Linear(3 -> 3)
        |          |      (2): nn.AddConstant
        |          |      (3): nn.Sigmoid
        |          |      (4): nn.ConcatTable {
        |          |        input
        |          |          |`-> (1): nn.Identity
        |          |          |`-> (2): nn.Sequential {
        |          |          |      [input -> (1) -> (2) -> output]
        |          |          |      (1): nn.MulConstant
        |          |          |      (2): nn.AddConstant
        |          |          |    }
        |          |           ... -> output
        |          |      }
        |          |    }
        |          |`-> (2): nn.Sequential {
        |          |      [input -> (1) -> (2) -> output]
        |          |      (1): nn.Linear(3 -> 3)
        |          |      (2): nn.Tanh
        |          |    }
        |           ... -> output
        |      }
        |    }
         ... -> output
    }
    (2): nn.FlattenTable
    (3): nn.ConcatTable {
      input
        |`-> (1): nn.Sequential {
        |      [input -> (1) -> (2) -> output]
        |      (1): nn.ConcatTable {
        |        input
        |          |`-> (1): nn.SelectTable
        |          |`-> (2): nn.SelectTable
        |           ... -> output
        |      }
        |      (2): nn.CMulTable
        |    }
        |`-> (2): nn.Sequential {
        |      [input -> (1) -> (2) -> output]
        |      (1): nn.ConcatTable {
        |        input
        |          |`-> (1): nn.SelectTable
        |          |`-> (2): nn.SelectTable
        |           ... -> output
        |      }
        |      (2): nn.CMulTable
        |    }
         ... -> output
    }
    (4): nn.CAddTable
  }
}
Checking 24 parameters
Test 1: 1024/1024 failed!
Test 2: 1024/1024 failed!
Test 3: 1024/1024 failed!
Test 4: 1024/1024 failed!
Test 5: 1024/1024 failed!
Test 6: 1024/1024 failed!
Test 7: 1024/1024 failed!
Test 8: 1024/1024 failed!
Test 9: 1024/1024 failed!
Test 10: 1024/1024 failed!
Test 11: 1024/1024 failed!
Test 12: 1024/1024 failed!
Test 13: 1024/1024 failed!
Test 14: 1024/1024 failed!
Test 15: 1024/1024 failed!
Test 16: 1024/1024 failed!
]=]
