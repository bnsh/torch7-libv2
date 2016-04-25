#! /usr/local/torch/install/bin/th

require "nn"
require "HighwayLayer"

-- Below is from Yoon Kim's HighwayMLP: https://github.com/yoonkim/lstm-char-cnn/blob/master/model/HighwayMLP.lua
require "nngraph"

local HighwayMLP = {}

function HighwayMLP.mlp(size, num_layers, bias, f)
    -- size = dimensionality of inputs
    -- num_layers = number of hidden layers (default = 1)
    -- bias = bias for transform gate (default = -2)
    -- f = non-linearity (default = ReLU)
    
    local output, transform_gate, carry_gate
    local num_layers = num_layers or 1
    local bias = bias or -2
    local f = f or nn.ReLU()
    local input = nn.Identity()()
    local inputs = {[1]=input}
    for i = 1, num_layers do        
        output = f(nn.Linear(size, size)(inputs[i]))
        transform_gate = nn.Sigmoid()(nn.AddConstant(bias)(nn.Linear(size, size)(inputs[i])))
        carry_gate = nn.AddConstant(1)(nn.MulConstant(-1)(transform_gate))
	output = nn.CAddTable()({
	       nn.CMulTable()({transform_gate, output}),
	       nn.CMulTable()({carry_gate, inputs[i]})	})
	table.insert(inputs, output)
    end
    return nn.gModule({input},{output})
end

-- Above is from Yoon Kim's HighwayMLP: https://github.com/yoonkim/lstm-char-cnn/blob/master/model/HighwayMLP.lua

local function main(argv)
	local numlayers = 3
	local sz = 3
	local bias = 0
	local bb = nn.Sequential()
	bb:add(nn.HighwayLayer(numlayers, sz,nn.Tanh(),bias,0.0))
	print(bb)

	local yk = HighwayMLP.mlp(sz,numlayers,bias,nn.Tanh())

	local bbp, bbg = bb:getParameters()
	local ykp, ykg = yk:getParameters()
	assert(ykp:nDimension() == 1)
	assert(ykp:nDimension() == bbp:nDimension())
	assert(ykp:size(1) == bbp:size(1))
	io.stderr:write(string.format("Checking %d parameters\n", bbp:size(1)))

	for i = 1,bbp:size(1)
	do
		bbp:zero()
		bbp[i] = 1
		local input = torch.randn(256,sz)
		local target = bb:forward(input)
		local candidates = { }
		for j = 1, ykp:size(1)
		do
			ykp:zero()
			ykp[j] = 1
			local output = yk:forward(input)
			local squareddeviations = torch.sum(torch.pow(torch.add(target, torch.mul(output,-1)),2))
			if squareddeviations == 0
			then
				table.insert(candidates, j)
			end
		end
		if #candidates > 1
		then
			io.stderr:write(string.format("parameter %d matches multiple parameters on yoon kim's version: [", i))
			for j = 1, #candidates
			do
				if j > 1
				then
					io.stderr:write(", ")
				end
				io.stderr:write(string.format("%d", candidates[j]))
			end
			io.stderr:write("]\n")
		elseif #candidates < 1
		then
			io.stderr:write(string.format("parameter %d doesn't match any parameters on yoon kim's version!\n", i))
		elseif candidates[1] ~= i
		then
			io.stderr:write(string.format("parameter %d is mismatched: matches parameter %d on yoon kim's version!\n", i, candidates[1]))
		else
			io.stderr:write(string.format("parameter %d check.\n", i))
		end
	end

	for i = 1, 16
	do
		params = torch.randn(bbp:size(1))
		bbp:copy(params)
		ykp:copy(params)
		local iters = 1024
		local rows = 1
		local failure = 0

		local bbout, ykout = nil, nil

		for j = 1, iters
		do
			data = torch.randn(rows, sz)
			bbout = bb:forward(data)
			ykout = yk:forward(data)

			local squareddeviations = torch.sum(torch.pow(torch.add(bbout, torch.mul(ykout,-1)),2))
			if squareddeviations ~= 0
			then
				failure = failure + 1
			end
		end
		if failure ~= 0
		then
			io.stderr:write(string.format("Test %d: %d/%d failed!\n", i, failure, iters))
		else
			io.stderr:write(string.format("Test %d: success.\n", i))
		end
	end
end

main(arg)
