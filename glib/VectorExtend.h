#pragma once
#include <iostream>
#include <vector>
#include <limits.h>
#include <algorithm>
using namespace std;
double VectorMean(vector<double>& dat);
double VectorMaximum(vector<double>& dat);
double VectorMinimum(vector<double>& dat);
double VectorStd(vector<double>& dat);
double VectorQuantile(vector<double>& dat,double p);
double VectorSum(vector<double>& dat);
int VectorIQR(vector<double>& dat);