function ByteTensorToStringTable(bt) 
	local tbl = bt:totable()
	for i, bytes in ipairs(tbl) do
		local str = string.char(unpack(bytes))
		local zero = string.find(str, '\0')
		if zero ~= nil then
			str = string.sub(str, 1, zero-1)
		end
		tbl[i] = str
	end
	return tbl
end
