#!  /usr/local/torch/install/bin/th

function fprintf(fh, fmt, ...)
	fh:write(string.format(fmt, unpack({...})))
end
