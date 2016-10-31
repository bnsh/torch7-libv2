extern "C" {
#include "lauxlib.h"
#include <luaT.h>
}

#include "pipeline.H"
#include "pipeline_impl.H"

extern "C" int luaopen_pipeline(lua_State *L);

static const char *metatablename = "x21.pipeline";

static inline void luax21_assert(lua_State *L, int predicate, const char *fmt, ...) __attribute__ ((format (printf, 3, 4)));
static inline void luax21_assert(lua_State *L, int predicate, const char *fmt, ...) {
	if (!predicate) {
		va_list argp;
		va_start(argp, fmt);
		char buffer[4096];
		vsnprintf(buffer, 4096, fmt, argp);
		va_end(argp);
		luaL_error(L, buffer);
	}
}

pipeline::pipeline() : _impl(new pipeline_impl()), n(0) { }

pipeline::~pipeline() { delete _impl; _impl = NULL; }

void pipeline::operator()(lua_State *L, int random, const THCudaTensor *inputbatch, THCudaTensor *outputbatch) {
	(*_impl)(L, random, inputbatch, outputbatch);
}

static int create_pipeline(lua_State *L) {
	pipeline **ptr = (pipeline **)lua_newuserdata(L, sizeof(pipeline *)); (*ptr) = NULL;
	luaL_getmetatable(L, metatablename);
	lua_setmetatable(L, -2);
	(*ptr) = new pipeline();
	return 1;
}

static int delete_pipeline(lua_State *L) {
	pipeline **ptr = (pipeline **)luaL_checkudata(L, 1, metatablename);
	if ((*ptr)) delete (*ptr); (*ptr) = NULL;
	return 0;
}

static int debug_pipeline(lua_State *L) {
	pipeline **ptr = (pipeline **)luaL_checkudata(L, 1, metatablename);
	luax21_assert(L, ((*ptr) != NULL), "This pipeline is invalid. Actually, how the fuck did this happen?!");
	(*ptr)->debug();

	return 0;
}


static int run_pipeline(lua_State *L) {
	pipeline **ptr = (pipeline **)luaL_checkudata(L, 1, metatablename);
	luax21_assert(L, ((*ptr) != NULL), "This pipeline is invalid. Actually, how the fuck did this happen?!");
	int random = lua_toboolean(L, 2);
	const THCudaTensor *input = (const THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");

	(*ptr)->operator()(L, random, input, output);

	return 0;
}

static const struct luaL_reg pipeline_static_methods[] = {
	{ "new", create_pipeline },
	{ NULL, NULL }
};

static const struct luaL_reg pipeline_methods[] = {
	{ "__gc", delete_pipeline },
	{ "run", run_pipeline },
	{ "debug", debug_pipeline },
	{ NULL, NULL }
};

int luaopen_pipeline(lua_State *L) {
	luaL_newmetatable(L, metatablename);

	lua_pushvalue(L, -1);
	lua_setfield(L, -2, "__index");

	luaL_register(L, NULL, pipeline_methods);
	luaL_register(L, "pipeline", pipeline_static_methods);
	return 1;
}
