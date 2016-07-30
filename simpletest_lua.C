#include <stdio.h>
#include <string.h>
#include <assert.h>
extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
#include <luaT.h>
}
#include <TH.h>
#include <THC.h>

extern "C" {
	int luaopen_simpletest(lua_State *L);
}

static int lua_simpletest_mul(lua_State *L) {
	const char *torch_DoubleTensor_id = luaT_typenameid(L, "torch.DoubleTensor");
	fprintf(stderr, "%s: %d: %s\n", __FILE__, __LINE__, torch_DoubleTensor_id);
	THDoubleTensor *t = (THDoubleTensor *)luaT_checkudata(L, 1, torch_DoubleTensor_id);
	THDoubleTensor_zero(t);
	THDoubleTensor_add(t, t, 0.01);
	

	return 0;
}

static int lua_simpletest_fill(lua_State *L) {
	int rows = lua_tointeger(L, 1);
	int columns = lua_tointeger(L, 2);

	THIntTensor *it = THIntTensor_newWithSize2d(rows, columns);
	THIntTensor_zero(it);
	luaT_pushudata(L, it, "torch.IntTensor");
	int id = 0;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < columns; ++j) {
			int idx = i * it->stride[0] + j;
			it->storage->data[idx] = id;
			id = id + 1;
		}
	}
	return 1;
}

static int lua_simpletest_nulltest(lua_State *L) {
	const char *torch_DoubleTensor_id = luaT_typenameid(L, "torch.DoubleTensor");
	if (lua_isnil(L, 1)) fprintf(stderr, "It's nil, and that's A. OK!\n");
	else luaT_checkudata(L, 1, torch_DoubleTensor_id);

	return 0;
}

static int lua_simpletest_tablefieldtest(lua_State *L) {
	lua_istable(L, 1);
	const char *field = luaL_checkstring(L, 2);

	lua_getfield(L, -2, field);
	int gt = lua_gettop(L);
	lua_pushinteger(L, gt);

	return 2;
}

static int lua_simpletest_stridetest(lua_State *L) {
	const char *torch_DoubleTensor_id = luaT_typenameid(L, "torch.DoubleTensor");
	THDoubleTensor *t = (THDoubleTensor *)luaT_checkudata(L, 1, torch_DoubleTensor_id);

	lua_newtable(L); // So, now the top entry is a table.
	lua_pushstring(L, "nDimension");
	lua_pushnumber(L, t->nDimension);
	lua_rawset(L, -3);
	lua_pushstring(L, "size");
	lua_newtable(L);
	for (int i = 0; i < t->nDimension; ++i) {
		lua_pushnumber(L, t->size[i]);
		lua_rawseti(L, -2, 1+i);
	}
	lua_rawset(L, -3);
	lua_pushstring(L, "stride");
	lua_newtable(L);
	for (int i = 0; i < t->nDimension; ++i) {
		lua_pushnumber(L, t->stride[i]);
		lua_rawseti(L, -2, 1+i);
	}
	lua_rawset(L, -3);
	return 1;
}

static const struct luaL_reg simpletest[] = {
	{"mul", lua_simpletest_mul },
	{"nulltest", lua_simpletest_nulltest },
	{"fill", lua_simpletest_fill },
	{"tablefieldtest", lua_simpletest_tablefieldtest },
	{"stridestest", lua_simpletest_stridetest },
	{NULL, NULL}
};

int luaopen_simpletest(lua_State *L) {
	luaL_openlib(L, "simpletest", simpletest, 0);
	lua_setglobal(L, "simpletest");
	return(0);
}
