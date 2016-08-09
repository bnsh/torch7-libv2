#! /usr/local/src/torch-2015-05-25/install/bin/th

require "fprintf"

local function dump_tensor(fh, indent, tensor)
	local st = string.gsub("\n" .. tensor:__tostring(), "\n", "\n"..indent)
	fprintf(fh, "%s", st)
end

function dump_table(fh, indent, t, dump_tensors)
-- We assume that the caller has left us in the properly indented position,
-- and that any succeeding newlines are also done by the caller.
	if (t == nil) then
		fprintf(fh, "nil")
	elseif (torch.isTensor(t)) then
		fprintf(fh, "%s(", torch.type(t))
		for i=1,t:dim() do
			if i > 1 then fprintf(fh, ",") end
			fprintf(fh, "%d", t:size(i))
		end
		fprintf(fh, ")")
		if dump_tensors then
			dump_tensor(fh, indent.."\t", t)
		end
	elseif (type(t) == "boolean") then
		fprintf(fh, "%s", tostring(t))
	elseif (type(t) == "function") then
		fprintf(fh, "%s", tostring(t))
	elseif (type(t) == "string") then
		fprintf(fh, "%s", t)
	elseif (type(t) == "number") then
		fprintf(fh, "%.7g", t)
	elseif (type(t) == "table") then
		local first = true
		for k, v in pairs(t) do
			if first then fprintf(fh, "{") else fprintf(fh, ",") end
			fprintf(fh, "\n%s	%s: ", indent, tostring(k))
			dump_table(fh, indent.."\t", v, dump_tensors)
			first = false
		end
		if first then
			fprintf(fh, "{ }", indent)
		else
			fprintf(fh, "\n%s}", indent)
		end
	else
		fprintf(fh, "{Don't know how to handle %s: %s}", torch.type(t), tostring(t))
	end
end
