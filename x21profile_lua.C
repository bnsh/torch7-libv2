#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <assert.h>
extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
#include <luaT.h>
}
#include <TH.h>
#include <THC.h>
#include <string>
#include <stack>
#include <map>
#include <deque>

using namespace std;

extern "C" {
	int luaopen_x21profile(lua_State *L);
}

static int profiling = 0;

class profile_info {
	public:
		profile_info(lua_Debug *ar);

		const string& id() const { return(_id); }
		double tod() const { return(_tod); }

		void dump(FILE *fp) const;
	private:
		string _id;
		int _event;
		const string _name;
		const string _namewhat;
		const string _what;
		const string _source;
		int _currentline;
		int _nups;
		int _linedefined;
		int _lastlinedefined;
		const string _short_src;
		double _tod;
};

profile_info::profile_info(lua_Debug *ar) :
	_event(ar->event),
	_name(ar->name ? ar->name : "NULL"),
	_namewhat(ar->namewhat ? ar->namewhat : "NULL"),
	_what(ar->what ? ar->what : "NULL"),
	_source(ar->source ? ar->source : "NULL"),
	_currentline(ar->currentline),
	_nups(ar->nups),
	_linedefined(ar->linedefined),
	_lastlinedefined(ar->lastlinedefined),
	_short_src(ar->short_src) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	_tod = tv.tv_sec + tv.tv_usec / 1000000.0;
	char buffer[65536];
	snprintf(buffer, 65535, "%s:%s:%d:%d", _source.c_str(), _name.c_str(), _linedefined, _lastlinedefined);
	_id = buffer;
}

void profile_info::dump(FILE *fp) const {
	if (_what == "Lua") {
		const char *event_type = NULL;
		if (_event == LUA_HOOKCALL) event_type = "LUA_HOOKCALL";
		else if (_event == LUA_HOOKRET) event_type = "LUA_HOOKRET";
		else if (_event == LUA_HOOKTAILRET) event_type = "LUA_HOOKTAILRET";
		else if (_event == LUA_HOOKLINE) event_type = "LUA_HOOKLINE";
		else if (_event == LUA_HOOKCOUNT) event_type = "LUA_HOOKCOUNT";

		if (event_type != NULL) fprintf(fp, "%.7f: %s: %s\n", _tod, _id.c_str(), event_type);
		else fprintf(fp, "%.7f: %s: UNKNOWN EVENT_TYPE: %d\n", _tod, _id.c_str(), _event);
		fprintf(fp, "{");
		fprintf(fp, "\n	event: %d", _event);
		fprintf(fp, "\n	name: %s", _name.c_str());
		fprintf(fp, "\n	namewhat: %s", _namewhat.c_str());
		fprintf(fp, "\n	what: %s", _what.c_str());
		fprintf(fp, "\n	source: %s", _source.c_str());
		fprintf(fp, "\n	currentline: %d", _currentline);
		fprintf(fp, "\n	nups: %d", _nups);
		fprintf(fp, "\n	linedefined: %d", _linedefined);
		fprintf(fp, "\n	lastlinedefined: %d", _lastlinedefined);
		fprintf(fp, "\n	short_src: %s", _short_src.c_str());
		fprintf(fp, "\n	tod: %.7f", _tod);
		fprintf(fp, "\n}\n");
	}
}

static stack<profile_info> callstack;
static map<string, deque<pair<double, double> > > timinginfo;

void profiler_hook(lua_State *L, lua_Debug *ar) {
	lua_getinfo(L, "nSlu", ar);
	profile_info p(ar);
	if (ar->event == LUA_HOOKCALL) callstack.push(p);
	else if (ar->event == LUA_HOOKRET) {
		while (!callstack.empty()) {
			const profile_info& top = callstack.top();
			timinginfo[top.id()].push_back(pair<double, double>(top.tod(), p.tod()));
			callstack.pop();

			if (top.id() == p.id()) break;
		}
	}
	else p.dump(stderr);
}

static const char *outputfn = NULL;
static int lua_x21profile_start(lua_State *L) {
	if (!profiling) {
		outputfn = luaL_checkstring(L, 1);
		fprintf(stderr, "%s: %d: Profiler initiated: will write to \"%s\"\n", __FILE__, __LINE__, outputfn);
		timinginfo.clear();
		while (!callstack.empty()) callstack.pop();
		profiling = 1;
		lua_sethook(L, profiler_hook, LUA_MASKCALL | LUA_MASKRET, -1); // -1 is count, which is only used for LUA_MASKCOUNT, which we are not using.
	}
	return 0;
}

static int lua_x21profile_stop(lua_State *L) {
	if (profiling) {
		fprintf(stderr, "%s: %d: Profiler exited: wrote to \"%s\"\n", __FILE__, __LINE__, outputfn);
		lua_sethook(L, profiler_hook, 0, -1);

// Actually, first let's find the minimum.
		double start = -1.0;
		for (map<string, deque<pair<double, double> > >::iterator i = timinginfo.begin(); i != timinginfo.end(); ++i) {
			for (deque<pair<double, double> >::iterator j = (*i).second.begin(); j != (*i).second.end(); ++j) {
				double l = (*j).first;
				if ((*j).first > (*j).second) l = (*j).second;
				if ((start == -1) || (l < start)) start = l;
			}
		}
		FILE *fp = fopen(outputfn, "w");
		if (fp) {
			fprintf(fp, "{");
			int firststring = 1;
			for (map<string, deque<pair<double, double> > >::iterator i = timinginfo.begin(); i != timinginfo.end(); ++i) {
				if (!firststring) fprintf(fp, ",");
				firststring = 0;
				fprintf(fp, "\n	\"%s\": [", (*i).first.c_str());

				int firsttime = 1;
				for (deque<pair<double, double> >::iterator j = (*i).second.begin(); j != (*i).second.end(); ++j) {
					if (!firsttime) fprintf(fp, ",");
					firsttime = 0;

					fprintf(fp, "\n		[ %.7f, %.7f, %.7f ]", (*j).first-start, (*j).second-start, (*j).second-(*j).first);
				}
				fprintf(fp, "\n	]");
			}
			fprintf(fp, "\n}");
		}

		profiling = 0;
		outputfn = NULL;
	}
	return 0;
}

static const struct luaL_reg x21profile[] = {
	{"start", lua_x21profile_start},
	{"stop", lua_x21profile_stop},
	{NULL, NULL}
};

int luaopen_x21profile(lua_State *L) {
	luaL_openlib(L, "x21profile", x21profile, 0);
	return(1);
}
