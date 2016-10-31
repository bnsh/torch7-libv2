#!  /usr/local/torch/install/bin/th

require "cutorch"
require "image"
require "lfs"
require "pipeline"

local function traverse(fn, pngs)
	print(fn)
	local ll = lfs.attributes(fn)
	if ll.mode == "directory" then
		local children = { }
		for child in lfs.dir(fn) do
			if child ~= "." and child ~= ".." then
				local cfn = string.format("%s/%s", fn, child)
				table.insert(children, cfn)
			end
		end
		for i, cfn in ipairs(children) do
			traverse(cfn, pngs)
		end
	elseif fn:sub(-4) == ".png" then
		table.insert(pngs, fn)
	end
end

function main(argv)
	local input = nil
	local output = nil

	local files = { }
	for i, fn in ipairs(argv) do
		traverse(string.format("%s/input", fn), files)
	end
	print(files)

	for i, fn in ipairs(files) do
		local img = image.load(fn, "float")
		if input == nil then
			input = torch.CudaTensor(#files, img:size(1), img:size(2), img:size(3))
		end
		input[{i,{},{},{}}]:copy(img)
	end
	input = input:permute(1, 3, 4, 2):contiguous()
	output = torch.CudaTensor(input:size(1), 227, 227, 3):fill(1.0):contiguous()

	local pp = pipeline.new()

	local tt = torch.Timer()
	pp:run(true, input, output)
	print(tt:time())

	output = output:permute(1,4,2,3)
	for i, fn in ipairs(files) do
		local ofn, _ = fn:gsub("/input/", "/output/")
		image.save(ofn, output[i])
	end
end

main({"/tmp/qball"})

collectgarbage()
