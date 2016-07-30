require "image"
require "fprintf"

local OverSample, parent = torch.class('nn.OverSample', 'nn.Module')

--[=[
All this module does, is simply take as input an image, and output
10 tables: one for each corner, one for the center, and one of each of the
same only flipped horizontally. Why? So as to better simulate the
test condition on AlexNet

I ASSUME batches!
]=]

function OverSample:__init(patchszx, patchszy)
	parent.__init(self)
	self.gradInput = nil
	self.output = nil
	self.patchszx = patchszx
	self.patchszy = patchszy
end

function OverSample:updateOutput(input)
assert(not input:ne(input):any())
	local batchsz = input:size(1)
	local channels = input:size(2)
	local szy = input:size(3)
	local szx = input:size(4)
	local patchszx = (self.patchszx < szx) and self.patchszx or szx
	local patchszy = (self.patchszy < szy) and self.patchszy or szy
	self.output = self.output or { }
	for i = 1, 10 do
		self.output[i] = self.output[i] or torch.Tensor():typeAs(input)
		self.output[i]:resize(batchsz, channels, patchszy, patchszx):zero()
	end
	for j = 1, batchsz do
		self.output[ 1][{j,{},{},{}}]:copy(image.crop(input[j]:double(), "tl", patchszx, patchszy))
		self.output[ 2][{j,{},{},{}}]:copy(image.crop(input[j]:double(), "tr", patchszx, patchszy))
		self.output[ 3][{j,{},{},{}}]:copy(image.crop(input[j]:double(), "bl", patchszx, patchszy))
		self.output[ 4][{j,{},{},{}}]:copy(image.crop(input[j]:double(), "br", patchszx, patchszy))
-- So, center seems to crop like so:
-- start = 1+math.floor((sz-patchsz)/2)
-- crop {start, start+patchsz-1}
		self.output[ 5][{j,{},{},{}}]:copy(image.crop(input[j]:double(),  "c", patchszx, patchszy))
		for k = 6, 10 do
			self.output[ k][{j,{},{},{}}]:copy(image.hflip(self.output[k-5][j]:double()))
		end
	end
	for i = 1, 10 do
		assert(not self.output[i]:ne(self.output[i]):any())
	end

	return self.output
end

function OverSample:updateGradInput(input, gradOutput)
	local batchsz = input:size(1)
	local channels = input:size(2)
	local szy = input:size(3)
	local szx = input:size(4)
	local patchszx = (self.patchszx < szx) and self.patchszx or szx
	local patchszy = (self.patchszy < szy) and self.patchszy or szy
	self.gradInput = self.gradInput or torch.Tensor(0):typeAs(input)

	self.gradInput:resizeAs(input)
-- OK, so we should be getting a table of ten tensors for gradOutput.
-- We just need to smash them into gradInput.
	self.gradInput:zero()
	for j = 1, batchsz do
		self.gradInput[{j,{},{1,patchszy},{1,patchszx}}]:add(gradOutput[1][j])
		self.gradInput[{j,{},{1,patchszy},{szx+1-patchszx,szx}}]:add(gradOutput[2][j])
		self.gradInput[{j,{},{szy+1-patchszy, szy},{1,patchszx}}]:add(gradOutput[3][j])
		self.gradInput[{j,{},{szy+1-patchszy, szy},{szx+1-patchszx,szx}}]:add(gradOutput[4][j])
		local startx = 1+math.floor((szx - patchszx)/2)
		local starty = 1+math.floor((szy - patchszy)/2)
		self.gradInput[{j,{},{starty,starty+patchszy-1},{startx,startx+patchszx-1}}]:add(gradOutput[5][j])

-- Now for the reverses.
		self.gradInput[{j,{},{1,patchszy},{1,patchszx}}]:add(image.hflip(gradOutput[1][j]:double()):cuda())
		self.gradInput[{j,{},{1,patchszy},{szx+1-patchszx,szx}}]:add(image.hflip(gradOutput[2][j]:double()):cuda())
		self.gradInput[{j,{},{szy+1-patchszy, szy},{1,patchszx}}]:add(image.hflip(gradOutput[3][j]:double()):cuda())
		self.gradInput[{j,{},{szy+1-patchszy, szy},{szx+1-patchszx,szx}}]:add(image.hflip(gradOutput[4][j]:double()):cuda())
		local startx = 1+math.floor((szx - patchszx)/2)
		local starty = 1+math.floor((szy - patchszy)/2)
		self.gradInput[{j,{},{starty,starty+patchszy-1},{startx,startx+patchszx-1}}]:add(image.hflip(gradOutput[5][j]:double()):cuda())
	end

	return self.gradInput
end
