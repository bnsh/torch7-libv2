#ifndef PIPELINE_H
#define PIPELINE_H

extern "C" {
#include <lua.h>
}
#include <THC.h>

/*
 * This class will take a full set of pictures all in gpu space
 * and perform a bunch of image operations on it.
 * The images are expected to be in
 * Y, X, C format ("packed channel" format)
 * and will be output in the same way. This means
 * ultimately, the Lua call will first have to call
 * 	:permute(3,1,2) and then
 *      :permute(2,3,1)
 */

class pipeline_impl;

class pipeline {
	public:
		pipeline();
		~pipeline();

		void operator()(lua_State *L, int random, const THCudaTensor *inputbatch, THCudaTensor *outputbatch);
		void debug() { fprintf(stderr, "%d\n",n++); }
	private:
		pipeline_impl *_impl;

		pipeline(const pipeline&);
		pipeline& operator=(const pipeline&);
		int n;
};

#endif
