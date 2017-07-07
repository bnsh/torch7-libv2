#! /usr/local/torch/install/bin/th

--[=[
	Basically, this is so that we'll have the same random data
	when testing between luatorch and pytorch.
--]=]

require "no_globals"
require "fprintf"
require "nnio"

local function main(args)
	if #args ~= 2 then
		print(args)
		fprintf(io.stderr, "Usage: th random_source_create.lua /tmp/random.bin 16777216\n")
		fprintf(io.stderr, "will generate a random binary file of size 16777216 doubles")
	else
		local filename = args[1]
		local size = tonumber(args[2])

		local random_data = torch.randn(size):double()
		nnio.store(filename, filename .. '-tmp', random_data)
		fprintf(io.stderr, "OK. %d doubles have been created in %s\n", size, filename, 'random_source_create.lua')
	end
end

no_globals(main)
