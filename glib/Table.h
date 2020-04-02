#pragma once
#include <vector>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <string>
#include "Statistics.h"
#include "PCLExtend.h"
#include "VectorExtend.h"
#include "Color.h"
using namespace std;
class Rrd
{
	public:
		int id_;
		double item1_;
		Rrd(){}
		Rrd(int id,double item1){
			id_=id;
			item1_=item1;
		}
};

class Table
{
	public:
		vector<Rrd> records_;
		vector<Rrd> records_sorted_;
		vector<int> records_hist_;
		double mean_,sigma_;		
		double min_=INT_MAX;
		double max_=-INT_MAX;

		// use on hierarchical method
		vector<int> believable_idx_;	// believable area index for upper level
		vector<int> unbelievable_idx_;	// unbelivable area index for upper level
		vector<int> lower_unbelievable_idx_;	// index for outlier below lower bound
		vector<int> upper_unbelievable_idx_;	// index for outlier greater than upper bound

		vector<V3> color_;
		

		double Q1_,Q3_;
		double median_;	
		double upper_inner_limit_,lower_inner_limit_;
		int flag_is_active_=0;

		// Quantile
		double Quantile(double p);
		double Median();
		double Mean();
		double Std();
		double Minimum();
		double Maximum();

		// 
		void EnableActive();
		void DisableActive();
		int Rows();
		void Resize(int n);
		void push_back(Rrd e);
		void Ascending(int item_index=1);
		void Descending(int item_index=1);
		void GetMeanAndVariance(int item_index=1);
		void GetMinimumAndMaximum(int item_index=1);
		void GetBoxplot(double lamda=20.0,int item_index=1);
		void SortBackup(int item_index=1);		
		void Print();
		double PDF(double t,double n,double h);
		double WritePDF();
		double ReversePDF(double P);
		void GetSpecifiedIndex(vector<int>& out,string str=">",double thresh1=0, double thresh2=0);

		// Standarize with Z-score method
		void Standardize_Zscore(int item_index=1);
		// Normalize with tanh
		void Normalize_Tanh(double scale=1.0, int item_index=1);		
		void Normalize_Min_Max(int item_index=1);
		// Write File
		void Write(string path);
		// 
		void MultiplyQ6(int m1,int m2,double scale);
		//
		void SetQ6(int m1,int m2);
		// Overturn the table
		void Overturn();
		// error
		void GetNormalDistributionError();
		//
		void LocalFilter(string str,pcl::PointCloud<PointType>::Ptr cloud,int K);
		void GetCorrespondingColor(int item_index=1);
		//
		void GetHistogram(int k=30);
		void GaussianOutlier(int k=1);

		void nPLOF(pcl::PointCloud<PointType>::Ptr cloud,int K);
		void SetActiveIndex(string erea_color, string status);



		// Get 
		vector<int> GetIndex(string str,double v);

		// Fast Remove 
		void FastRemove(int index);
		void FastRemove(vector<int> indices);
		
};