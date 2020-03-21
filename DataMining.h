#pragma once
#include <iostream>
#include <math.h>
#include "PCLExtend.h"
#include <vector>
#include "Table.h"
using namespace std;
class DataMining
{
	public:
		// Common Parameters		
		pcl::PointCloud<PointType>::Ptr cloud_;
		pcl::search::KdTree<PointType>::Ptr kdtree_;

		// LOOP Parameters
		vector<double> scores_;
		vector<double> sigma_;
		vector<double> plof_;

		Table rst_;

		// Internal Function
		double erf(double x);
		
		// External Function
		DataMining(pcl::PointCloud<PointType>::Ptr cloud);
		void LOOP(int k=31,double threshould=0.8);
};

void EntropyWithKDE(pcl::PointCloud<PointType>::Ptr cloud);