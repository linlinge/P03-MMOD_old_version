#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include "SignalProcessing.h"
using namespace std;
void DaubechiesWavelet(vector<double>& dat,vector<double>& output);
void DaubechiesWavelet(vector<float>& dat,vector<double>& output);