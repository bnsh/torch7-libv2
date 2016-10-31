#! /usr/local/torch/install/bin/th

require "cutorch"
require "image"
require "imageutils"
require "dump_table"
require "fprintf"

function testimage()
	local halfsz = 240
	local fullsz = halfsz * 2 + 1
	local img = torch.DoubleTensor(fullsz, fullsz,3):fill(255)

	for i = -10,10 do
		for j = -10,10 do
			img[{halfsz+i,halfsz+j,1}] = 255
			img[{halfsz+i,halfsz+j,2}] = 0
			img[{halfsz+i,halfsz+j,3}] = 0
		end
	end

	for j = 1, 20 do
		for i = 1, fullsz do
			img[{j, i, 1}] = 0
			img[{j, i, 2}] = 0
			img[{j, i, 3}] = 255
		end

		for i = 1, fullsz do
			img[{i, j, 1}] = 0
			img[{i, j, 2}] = 255
			img[{i, j, 3}] = 0
		end

		for i = 1, fullsz do
			img[{fullsz-j+1, i, 1}] = 0
			img[{fullsz-j+1, i, 2}] = 255
			img[{fullsz-j+1, i, 3}] = 255
		end

		for i = 1, fullsz do
			img[{i, fullsz-j+1, 1}] = 255
			img[{i, fullsz-j+1, 2}] = 255
			img[{i, fullsz-j+1, 3}] = 0
		end
	end

	return img
end

function gaussian_blur_test()
	local img = image.load("/tmp/imageutilstest/miles-gaussian_blur.png"):permute(2,3,1)
	for i = 1,128 do
		local halfsz = math.random(1,5)
		local sz = halfsz * 2 + 1
		local a = torch.uniform(0,1)
	--[=[
	a + (sz*sz-1)b == 1
	b = (1-a) / (sz*sz-1)
	]=]
		local b = (1-a) / (sz*sz - 1)
		local kernel = torch.CudaTensor(sz, sz):fill(b)
		kernel[{halfsz+1,halfsz+1}] = a

		qq = imageutils.gaussian_blur(kernel, img:cuda())
		image.save(string.format("/tmp/imageutilstest/miles-gaussian_blur-%03dx%d.png", i, halfsz), qq:permute(3,1,2))
	end
end

function rotatecrop_test()
	local img = image.load("/tmp/imageutilstest/leela-rotatecrop.png"):permute(2,3,1)
	-- local img = testimage()
	
	for i = 1,128 do
		local theta = (i-1) * 2 * math.pi / 128
		qq = imageutils.rotatecrop(theta, img:cuda())
		image.save(string.format("/tmp/imageutilstest/leela-rotatecrop-%03d.png", i), qq:permute(3,1,2))
	end
end

function scale_test()
	local img = image.load("/tmp/imageutilstest/leela-scale.png"):permute(2,3,1)
	-- local img = testimage()
	for i = 1,128 do
		fprintf(io.stderr, "%d\n", i)
		local sx = (0.5 + (i-1) / 127.) * img:size(2) -- This will go from 0.5..1.5
		local sy = (0.5 + (i-1) / 127.) * img:size(1) -- This will go from 0.5..1.5
		qq = imageutils.scale(sx, sy, img:cuda())
		image.save(string.format("/tmp/imageutilstest/leela-scale-%03d.png", i), qq:permute(3,1,2))
	end
end

function normalized_square_test()
	local img = image.load("/tmp/imageutilstest/mom_miles-normalized_square.png"):permute(2,3,1)
	-- local img = testimage()
	for i = 1,128 do
		fprintf(io.stderr, "%d\n", i)
		local sx = (0.5 + (i-1) / 127.) * img:size(2) -- This will go from 0.5..1.5
		local sy = (0.5 + (i-1) / 127.) * img:size(1) -- This will go from 0.5..1.5
		qq = imageutils.normalized_square(sy, img:cuda())
		image.save(string.format("/tmp/imageutilstest/mom_miles-normalized_square-%03d.png", i), qq:permute(3,1,2))
	end
end

function colorwash_test()
	local img = image.load("/tmp/imageutilstest/beena_binesh-colorwash.png"):permute(2,3,1)
	-- local img = testimage()
	
	for i = 1,128 do
		local lo = 0
		local hi = 1.5
		local p = lo + (hi-lo) * (i-1) / 127.0
		print(p)
		qq = imageutils.colorwash(p, img:cuda())
		image.save(string.format("/tmp/imageutilstest/beena_binesh-colorwash-%03d.png", i), qq:permute(3,1,2))
	end
end

function adjust_contrast_test()
	local img = image.load("/tmp/imageutilstest/beena_binesh_older-adjust_contrast.png"):permute(2,3,1)
	-- local img = testimage()
	
	for i = 1,128 do
		local lo = 0
		local hi = 1.5
		local contrast = lo + (hi-lo) * (i-1) / 127.0
		print(contrast)
		qq = imageutils.adjust_contrast(contrast, 1.0, img:cuda())
		image.save(string.format("/tmp/imageutilstest/beena_binesh_older-adjust_contrast-%03d.png", i), qq:permute(3,1,2))
	end
end

-- gaussian_blur_test()
-- rotatecrop_test()
-- scale_test()
normalized_square_test()
-- colorwash_test()
-- adjust_contrast_test()
