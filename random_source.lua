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

	BUT. We can use a seed, and put it in a consistent state.
	Now, I no longer need random_source_create.lua
]=]

local function random_source(seed, sz, generator)
	if seed == nil then seed = 12345 end
	if generator == nil then generator = torch.randn end
	torch.manualSeed(seed)
	local random_src = generator(sz)
	local pos = 0

	local function grab(size)
		local returnvalue = torch.FloatTensor(size):zero()
		local rpos = 0

		while rpos < size do
			local grabsz = size-rpos
			if pos + grabsz >= random_src:size(1) then
				grabsz = random_src:size(1) - pos
			end

			returnvalue:indexCopy(1, torch.range(rpos+1, rpos+grabsz):long(), random_src[{{pos+1, pos+grabsz}}]:float())

			pos = (pos + grabsz) % random_src:size(1)
			rpos = rpos + grabsz
		end
		return returnvalue
	end

	return grab
end

return random_source
