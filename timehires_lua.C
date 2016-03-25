#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
#include <luaT.h>
}
#include <TH.h>
#include <THC.h>
#include <sys/time.h>

extern "C" {
	int luaopen_timehires(lua_State *L);
}

static int lua_timehires(lua_State *L) {
	struct timeval tv;
	struct timezone tz;

	if (0 == gettimeofday(&tv, &tz)) {
		double d = tv.tv_sec + tv.tv_usec / 1000000.0;
		lua_pushnumber(L, d);
	}
	else lua_pushnil(L);
	return 1;
}

static const struct luaL_reg timehires[] = {
	{"timehires", lua_timehires},
	{NULL, NULL}
};

int luaopen_timehires(lua_State *L) {
	luaL_openlib(L, "timehires", timehires, 0);
	lua_getglobal(L, "os");
	lua_getfield(L, -2, "timehires");
	lua_setfield(L, -2, "timehires");
	lua_pop(L, 3);
	return(0);
}
