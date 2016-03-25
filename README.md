These are just a few torch tools I've made..

 * BinaryStochastic.lua will make a node that is basically a sigmoid internally, but will output either a 1 or a 0 stochastically with p=sigmoid(x)
 * HighwayLayer.lua is my attempt at writing a [highway layer](http://arxiv.org/abs/1505.00387)
 * TemporalBatchNormalization.lua will do a one dimensional normalization. It's really just a very very basic modification of an earlier version of SpatialBatchNormalization.lua . However looking at the newest version of SpatialBatchNormalization.lua, I wonder if it's now irrelevant.
 * no\_globals.lua ... Is just a function that helps me avoid accidentally creating globals. Odds are there's something that does this better.
 * timehires\_lua.C just gives a hires timer to lua. (It may be redundant, I just didn't know
   how to do it.
