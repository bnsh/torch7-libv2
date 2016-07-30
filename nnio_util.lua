#! /usr/local/torch/install/bin/th

require "nnio"
require "fprintf"
require "dump_table"

local nnio_util = { }

local function traverse(tbl, path, callback)
	if tbl.weight ~= nil
	then
		callback(tbl, path)
		collectgarbage()
	end
	if tbl.modules ~= nil
	then
		for idx, module in ipairs(tbl.modules)
		do
			table.insert(path, idx)

			traverse(module, path, callback)

			table.remove(path, #path)
		end
	end
end

local function prefixize(dn, pth)
	local prefix = dn .. "/layer"
	for depth, value in ipairs(pth)
	do
		prefix = prefix .. "-" .. tostring(value)
	end
	return prefix
end

function nnio_util.store(dn, mlp)
	lfs.mkdir(dn)
	local pth = { }
	local function lstore(tbl, pth)
		local prefix = prefixize(dn, pth)
		local data = nil

		if tbl.bias ~= nil
		then
			local w = tbl.weight:double()
			local b = tbl.bias:double()
			data = torch.cat(w:reshape(w:nElement()), b:reshape(b:nElement()))
		else
			data = tbl.weight:double():reshape(tbl.weight:nElement())
		end
		nnio.store(prefix .. ".bin", prefix .. "-tmp.bin", data, torch.type(tbl))
	end
	traverse(mlp, pth, lstore)
end

function nnio_util.restore(dn, mlp)
	local pth = { }
	local function lrestore(tbl, pth)
		local prefix = prefixize(dn, pth)
		local data = nil
		if tbl.bias ~= nil
		then
			local w = tbl.weight:double()
			local b = tbl.bias:double()
			data = torch.cat(w:reshape(w:nElement()), b:reshape(b:nElement()))
		else
			data = tbl.weight:double():reshape(tbl.weight:nElement())
		end
--[=[
		fprintf(io.stderr, "%s.bin: data=", prefix)
		dump_table(io.stderr, "", data); fprintf(io.stderr, " tbl="); dump_table(io.stderr, "", tbl); fprintf(io.stderr, "\n");
]=]
		local tp = nnio.load(prefix .. ".bin", data)
-- Now just shove it into the right spots.
		tbl.weight:copy(data[{{1,tbl.weight:nElement()}}])
		if tbl.bias ~= nil
		then
			local s = 1 + tbl.weight:nElement()
			local e = s + tbl.bias:nElement() - 1
			tbl.bias:copy(data[{{s, e}}])
		end
	end
	traverse(mlp, pth, lrestore)
end

return nnio_util
