#! /usr/local/torch/install/bin/th

require "cutorch"
require "image"
require "imageutils"
require "dump_table"

--[=[
1 2 3	1 2 3
1 3 2	1 3 2
2 1 3	2 1 3
2 3 1	3 1 2
3 1 2	2 3 1
3 2 1	3 2 1
]=]

local permutations = {
	{ {2,3,1}, {3,1,2} }
}


for i = 1,2 do
	local img = image.load("/tmp/pipeline/input/9780698153288-384.png"):permute(2,3,1)
	local tt = torch.Tensor():typeAs(img)
	local halfsz = math.random(5,5)
	local sz = halfsz * 2 + 1
	local a = torch.uniform(0,1)
--[=[
a + (sz*sz-1)b == 1
b = (1-a) / (sz*sz-1)
]=]
	local b = (1-a) / (sz*sz - 1)
	local kernel = torch.CudaTensor(sz, sz):fill(b)
	kernel[{halfsz+1,halfsz+1}] = a
	print(kernel:size())

	qq = imageutils.gaussian_blur(kernel, img:cuda())
	image.save(string.format("/tmp/pipeline/input/9780698153288-384-kerneltest-%03dx%d.png", i, halfsz), qq:permute(3,1,2))
end
