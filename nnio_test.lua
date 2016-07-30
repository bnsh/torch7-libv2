#!  /usr/local/torch/install/bin/th

require "nnio"
require "lfs"
require "no_globals"

function main(argv)
	if (#argv > 1)
	then
		local dir = argv[1]
		local iterations = tonumber(argv[2] or "128")
		lfs.mkdir(dir)

		for i = 1,iterations
		do
			local fn = string.format("%s/twod-%d.bin", dir, i)
			local fntmp = string.format("%s/twod-%d-tmp.bin", dir, i)
			local rows = math.floor(math.random() * 128) + 1
			local cols = math.floor(math.random() * 128) + 1
			local orig = torch.randn(rows, cols)
			nnio.store(fn, fntmp, orig, torch.type(orig))
			local reconstituted = torch.zeros(rows, cols)
			local tpread = nnio.load(fn, reconstituted)
			local diff = (reconstituted - orig)
			diff:cmul(diff)
			assert((diff:sum() == 0.0))
		end
		print(string.format("%d iterations passed for 2-d matrices.", iterations))
		for i = 1,iterations
		do
			local fn = string.format("%s/oned-%d.bin", dir, i)
			local fntmp = string.format("%s/oned-%d-tmp.bin", dir, i)
			local cols = math.floor(math.random() * 128) + 1
			local orig = torch.randn(cols)
			nnio.store(fn, fntmp, orig, torch.type(orig))
			local reconstituted = torch.zeros(cols)
			local tpread = nnio.load(fn, reconstituted)
			local diff = (reconstituted - orig)
			diff:cmul(diff)
			assert((diff:sum() == 0.0))
		end
		print(string.format("%d iterations passed for 1-d matrices.", iterations))
	else
		print([=[
Usage: th nnio_test.lua <directory> <iterations>

Tests nnio.load and nnio.store with random vectors, to see if they "work"..
Basically, is testing to see if what is stored comes back unmolested upon a load.
]=])
	end
end

no_globals(main)
