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
    [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
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
    (5): nn.ConcatTable {
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
    (6): nn.FlattenTable
    (7): nn.ConcatTable {
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
    (8): nn.CAddTable
    (9): nn.ConcatTable {
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
    (10): nn.FlattenTable
    (11): nn.ConcatTable {
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
    (12): nn.CAddTable
  }
}
Checking 72 parameters
parameter 1 check.
parameter 2 check.
parameter 3 check.
parameter 4 check.
parameter 5 check.
parameter 6 check.
parameter 7 check.
parameter 8 check.
parameter 9 check.
parameter 10 matches multiple parameters on yoon kim's version: [10, 34, 58]
parameter 11 matches multiple parameters on yoon kim's version: [11, 35, 59]
parameter 12 matches multiple parameters on yoon kim's version: [12, 36, 60]
parameter 13 check.
parameter 14 check.
parameter 15 check.
parameter 16 check.
parameter 17 check.
parameter 18 check.
parameter 19 check.
parameter 20 check.
parameter 21 check.
parameter 22 check.
parameter 23 check.
parameter 24 check.
parameter 25 check.
parameter 26 check.
parameter 27 check.
parameter 28 check.
parameter 29 check.
parameter 30 check.
parameter 31 check.
parameter 32 check.
parameter 33 check.
parameter 34 matches multiple parameters on yoon kim's version: [10, 34, 58]
parameter 35 matches multiple parameters on yoon kim's version: [11, 35, 59]
parameter 36 matches multiple parameters on yoon kim's version: [12, 36, 60]
parameter 37 check.
parameter 38 check.
parameter 39 check.
parameter 40 check.
parameter 41 check.
parameter 42 check.
parameter 43 check.
parameter 44 check.
parameter 45 check.
parameter 46 check.
parameter 47 check.
parameter 48 check.
parameter 49 check.
parameter 50 check.
parameter 51 check.
parameter 52 check.
parameter 53 check.
parameter 54 check.
parameter 55 check.
parameter 56 check.
parameter 57 check.
parameter 58 matches multiple parameters on yoon kim's version: [10, 34, 58]
parameter 59 matches multiple parameters on yoon kim's version: [11, 35, 59]
parameter 60 matches multiple parameters on yoon kim's version: [12, 36, 60]
parameter 61 check.
parameter 62 check.
parameter 63 check.
parameter 64 check.
parameter 65 check.
parameter 66 check.
parameter 67 check.
parameter 68 check.
parameter 69 check.
parameter 70 check.
parameter 71 check.
parameter 72 check.
Test 1: success.
Test 2: success.
Test 3: success.
Test 4: success.
Test 5: success.
Test 6: success.
Test 7: success.
Test 8: success.
Test 9: success.
Test 10: success.
Test 11: success.
Test 12: success.
Test 13: success.
Test 14: success.
Test 15: success.
Test 16: success.
]=]
