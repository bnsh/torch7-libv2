#! /usr/local/torch/install/bin/th

require "cutorch"
require "nn"
require "CMeanTable"

local function bernoullize(r, b)
	local c = r:clone()
	if b:any() then
		local bi = torch.range(1, r:size()[1]):long()[b]
		c:indexFill(1, bi, 0)
	end
	return c
end

local function constructTensor(p, sz)
-- We want a table that has a bunch of entries.
	local raw = torch.randn(sz)
	local zerothese = torch.ByteTensor(sz[1]):bernoulli(p)
	local result = bernoullize(raw, zerothese)
	return raw, zerothese, result
end

function dumpTensorContents(fh, t)
	fh:write(string.format("{", name))
	if torch.type(t) == "table" then
	else
		for i = 1, t:size(1) do
			if i > 1 then fh:write(",") end
			if t:dim() == 1 then
				fh:write(string.format("%.7f", t[i]))
			else
				fh:write("\n")
				dumpTensorContents(fh, t[i])
			end
		end
	end
	if t:dim() > 1 then fh:write("\n") end
	fh:write("}")
end

function dumpTensor(fh, t)
	local sz = t:size()
	fh:write(string.format("%s(", t:type()))
	dumpTensorContents(fh, t)
	fh:write(string.format(")"));
end

function dumpTable(fh, t)
	local first = true
	fh:write("{")
	for key, value in pairs(t) do
		if not first then
			fh:write(",")
		end
		first = false
		if torch.type(key) == "number" then
			fh:write(string.format("\n\t[%d]", key))
		else
			fh:write(string.format("\n\t\"%s\"", tostring(key)))
		end
		fh:write("=")
		dumpValue(fh, value)
	end
	fh:write("\n}")
end

function dumpValue(fh, t)
	if torch.isTensor(t) then dumpTensor(fh, t)
	elseif torch.type(t) == "table" then dumpTable(fh, t)
	elseif torch.type(t) == "number" then fh:write(t)
	elseif torch.type(t) == "string" then fh:write(string.format("[===[%s]===])", t))
	else fh:write("[====[%s]====]", tostring(t))
	end
end

function dump(fh, name, t)
	fh:write(name .. " = ")
	dumpValue(fh, t)
	fh:write("\n")
end

function single_round(p, batchsz, numtables, ...)
	local vecsz = torch.LongStorage({...})
	local rawdata = { }
	local bernoullis = { }
	local cookeddata = { }
	local target = torch.randn(vecsz)
	for i = 1, numtables do
		local r, b, c = constructTensor(p, vecsz)
		rawdata[i] = r
		bernoullis[i] = b
	end
	for i, k in ipairs(rawdata) do
		cookeddata[i] = bernoullize(rawdata[i], bernoullis[i])
	end

	local cmeantable = nn.CMeanTable()
	local mean = cmeantable:forward(cookeddata)
	local crit = nn.MSECriterion()
	local err = crit:forward(mean, target)
	local gradOutput = crit:backward(mean, target)
	local purportedGradInput = cmeantable:backward(cookeddata, gradOutput)
	local epsilon = 0.0001

	local expectedGradInput = { }
	local sumerrsquared = 0
	for i, _ in ipairs(rawdata)
	do
		local r = rawdata[i]
		local b = bernoullis[i]
		expectedGradInput[i] = torch.zeros(purportedGradInput[i]:size())
		local linearr = r:view(r:numel())
		local lineare = expectedGradInput[i]:view(expectedGradInput[i]:numel())
		local lineara = purportedGradInput[i]:view(purportedGradInput[i]:numel())
		for j = 1, linearr:numel() do
			local x = linearr[j]
			linearr[j] = x + epsilon
			cookeddata[i] = bernoullize(r, b)
			local errp1 = crit:forward(cmeantable:forward(cookeddata), target)

			linearr[j] = x - epsilon
			cookeddata[i] = bernoullize(r, b)
			local errm1 = crit:forward(cmeantable:forward(cookeddata), target)

			local f = (errp1 - errm1) / (2 * epsilon)
			lineare[j] = f
			linearr[j] = x;
			cookeddata[i] = bernoullize(r, b)
		end
		sumerrsquared = sumerrsquared + (expectedGradInput[i] - purportedGradInput[i]):pow(2):sum()
	end
	return (math.sqrt(sumerrsquared))
end
	
function test_cmeantable(iterations, p, batchsz, numtables,...)
	for i = 1, iterations do
		local rms = single_round(p, batchsz, numtables, ...)
		local status = "FAIL"
		if (rms < 1e-5) then status = "PASS" end
		print(string.format("%s %d: %.7g", status, i, rms))
		assert(status == "PASS")
	end
end

test_cmeantable(128,0.5,128,5,10)
