#pragma once

#include <stdint.h>
#include <windows.h>
#include <xtimec.h>

using namespace std;

double PCFreq = 0.0;
__int64 CounterStart = 0;

void StartCounter()
{
	uint64_t li = _Query_perf_frequency();

	PCFreq = double(li) / 1000.0;

	li = _Query_perf_counter();
	CounterStart = li;
}
double GetCounter()
{
	uint64_t li = _Query_perf_counter();
	return double(li - CounterStart) / PCFreq;
}