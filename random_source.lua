#! /usr/local/torch/install/bin/th

--[=[
	Basically, this is so that we'll have the same random data
	when testing between luatorch and pytorch. You create the
	random data with random_source_create.lua. I wonder though.
	Should I just use mersenne twister and be done with this??
	Actually, no, this is still better than that. Because, actually
	in point of fact, both pytorch and luatorch _do_ use mersenne
	twister. But, this allows me to exactly control where in
	the sequence things are when they get loaded.
]=]

local function random_source(fn, sz)
	local random_src = torch.DoubleTensor(sz)
	nnio.load(fn, random_src)
	local pos = 0

	local function grab(size)
		local returnvalue = torch.FloatTensor(size):zero()
		local rpos = 0

		while rpos < size do
			local grabsz = size
			if pos + grabsz >= random_src:size(1) then
				grabsz = random_src:size(1) - pos
			end

			-- Now, those are zero based (Because I
			-- originally wrote simile12fulltest.py)
			returnvalue:indexCopy(1, torch.range(rpos+1, rpos+grabsz):long(), random_src[{{pos+1, pos+grabsz}}]:float())

			pos = (pos + grabsz) % random_src:size(1)
			rpos = rpos + grabsz
		end
		return returnvalue
	end

	return grab
end

return random_source
