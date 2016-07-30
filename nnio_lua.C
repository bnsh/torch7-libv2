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

extern "C" {
	int luaopen_nnio(lua_State *L);
}

static void recursively_read(int& errs, int errsz, char *errstr, const char *fn, FILE *fp, THDoubleTensor *t, int curdim, double *base) {
	if (curdim == t->nDimension - 1) {
		if (t->stride[curdim] != 1) {
			errs++;
			snprintf(errstr, errsz, "%s: Final dimension stride is not 1 (%ld).", fn, t->stride[curdim]);
		}
		else {
			long sz = fread(base, sizeof((*base)), t->size[curdim], fp);
			if (sz != t->size[curdim]) {
				errs++;
				snprintf(errstr, errsz, "%s: Error reading/writing %ld doubles (only read/wrote %ld)", fn, t->size[curdim], sz);
			}
			
		}
	}
	else {
		for (int i = 0; (i < t->size[curdim]) && (errs == 0); ++i) {
			recursively_read(errs, errsz, errstr, fn, fp, t, 1+curdim, base + i * t->stride[curdim]);
		}
	}
}

static void recursively_write(int& errs, int errsz, char *errstr, const char *fn, FILE *fp, const THDoubleTensor *t, int curdim, const double *base) {
	if (curdim == t->nDimension - 1) {
		if (t->stride[curdim] != 1) {
			errs++;
			snprintf(errstr, errsz, "%s: Final dimension stride is not 1 (%ld).", fn, t->stride[curdim]);
		}
		else {
			long sz = fwrite(base, sizeof((*base)), t->size[curdim], fp);
			if (sz != t->size[curdim]) {
				errs++;
				snprintf(errstr, errsz, "%s: Error reading/writing %ld doubles (only read/wrote %ld)", fn, t->size[curdim], sz);
			}
			
		}
	}
	else {
		for (int i = 0; (i < t->size[curdim]) && (errs == 0); ++i) {
			recursively_write(errs, errsz, errstr, fn, fp, t, 1+curdim, base + i * t->stride[curdim]);
		}
	}
}

static int lua_nnio_load(lua_State *L) {
	const char *fn = luaL_checkstring(L, 1);
	THDoubleTensor *t = (THDoubleTensor *)luaT_checkudata(L, 2, "torch.DoubleTensor");
	int rv = 0;
	int errs = 0;
	char errstr[1024];

	FILE *fp = fopen(fn, "r");
	if (fp) {
		int nDimensionAssert;
		unsigned int tlen = 0;
		if (1 != fread(&tlen, sizeof(tlen), 1, fp)) {
			errs++;
			snprintf(errstr, 1024, "%s: Error reading type length", fn);
		}
		else {
			char *typestr = new char[tlen+1]; memset(typestr, '\0', tlen+1);
			if (tlen != fread(typestr, sizeof((*typestr)), tlen, fp)) {
				errs++;
				snprintf(errstr, 1024, "%s: Error reading type info", fn);
			}
			else if (1 != fread(&nDimensionAssert, sizeof(nDimensionAssert), 1, fp)) {
				errs++;
				snprintf(errstr, 1024, "%s: Error reading dimensionality", fn);
			}
			else if (nDimensionAssert != t->nDimension) {
				errs++;
				snprintf(errstr, 1024, "%s: Expected a pre-created table with %d dimensions, not %d", fn, nDimensionAssert, t->nDimension);
			}
			else {
				// Now we need to make an arbitrary sized array. *sigh*
				long int *sizeAssert = new long int[nDimensionAssert];
				long r = fread(sizeAssert, sizeof((*sizeAssert)), nDimensionAssert, fp);
				if (r != nDimensionAssert) {
					errs++;
					snprintf(errstr, 1024, "%s: Unable to read dimension data (expected size information on %d dimensions, only read %ld)", fn, nDimensionAssert, r);
				}
				else {
					// Verify that the dimensions are the same as what we expect.
					int samesizes = 1;
					for (int i = 0; i < nDimensionAssert; ++i) samesizes = samesizes && (t->size[i] == sizeAssert[i]);
					if (!samesizes) {
						errs++;

						char fromdisk[1024]; memset(fromdisk, '\0', 1024);
						char fromargs[1024]; memset(fromargs, '\0', 1024);
						char single[1024];

						for (int i = 0; i < nDimensionAssert; ++i) {
							if (i) {
								strcat(fromdisk, ", ");
								strcat(fromargs, ", ");
							}
							snprintf(single, 1024, "%ld", sizeAssert[i]);
							strcat(fromdisk, single);
							snprintf(single, 1024, "%ld", t->size[i]);
							strcat(fromargs, single);
						}

						snprintf(errstr, 1024, "%s: Incompatible tensor sizes: Expected [%s] found [%s]", fn, fromdisk, fromargs);
					}
					else {
						recursively_read(errs, 1024, errstr, fn, fp, t, 0, t->storage->data + t->storageOffset);
						if (errs == 0) {
							rv = 1;
							lua_pushlstring(L, typestr, strlen(typestr));
						}
					}
				}
				delete[] sizeAssert; sizeAssert = NULL;
			}
			if (typestr != NULL) delete[] typestr; typestr = NULL;
		}
		fclose(fp); fp = NULL;
	}
	else fprintf(stderr, "nnio.load failed! Error opening file %s for input.. Ignoring and continuing...\n", fn);

	if (errs) luaL_error(L, errstr);

	return rv;
}

static int lua_nnio_store(lua_State *L) {
	const char *fn = luaL_checkstring(L, 1);
	const char *fntmp = luaL_checkstring(L, 2);
	const THDoubleTensor *t = (const THDoubleTensor *)luaT_checkudata(L, 3, "torch.DoubleTensor");
	const char *type = luaL_checkstring(L, 4);
	unsigned int tlen = 1+strlen(type);

	FILE *fp = fopen(fntmp, "w");
	if (fp) {
		int errs = 0;
		char errstr[1024];
		if (
			(1 != fwrite(&tlen, sizeof(tlen), 1, fp)) ||
			(tlen != fwrite(type, sizeof((*type)), tlen, fp))
		) {
			snprintf(errstr, 1024, "%s: Error writing data type %s", fntmp, type);
			errs++;
		}
		if (
			(1 != fwrite(&t->nDimension, sizeof(t->nDimension), 1, fp)) ||
			((unsigned long int)t->nDimension != fwrite(t->size, sizeof(t->size[0]), t->nDimension, fp))
		) {
			snprintf(errstr, 1024, "%s: Error dimension information", fntmp);
			errs++;
		}
		else recursively_write(errs, 1024, errstr, fn, fp, t, 0, t->storage->data + t->storageOffset);

		fclose(fp); fp = NULL;
		if (errs == 0) rename(fntmp, fn);
		else luaL_error(L, errstr);
	}
	else luaL_error(L, "nnio.store failed! Error opening file %s for output", fntmp);

	return 0;
}

static const struct luaL_reg nnio[] = {
	{"store", lua_nnio_store },
	{"load", lua_nnio_load },
	{NULL, NULL}
};

int luaopen_nnio(lua_State *L) {
	luaL_openlib(L, "nnio", nnio, 0);
	lua_setglobal(L, "nnio");
	return(0);
}
