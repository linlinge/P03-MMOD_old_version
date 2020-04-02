#include "VectorExtend.h"
double VectorMean(vector<double>& dat)
{
    double rst=0;
    for(int i=0;i<dat.size();i++)
        rst+=dat[i];
    
    return rst/dat.size();
}

double VectorMinimum(vector<double>& dat)
{
    double rst=INT_MAX;
    for(int i=0;i<dat.size();i++)
        rst=rst<dat[i] ? rst:dat[i];
    return rst;
}
double VectorMaximum(vector<double>& dat)
{
    double rst=-INT_MAX;
    for(int i=0;i<dat.size();i++){
        rst=rst>dat[i] ? rst:dat[i];
    }
    return rst;
}
double VectorStd(vector<double>& dat)
{
    double rst=0;
    double dat_mean=VectorMean(dat);
    for(int i=0;i<dat.size();i++){
        rst+=(dat[i]-dat_mean)*(dat[i]-dat_mean);
    }
    return rst/(dat.size()-1);
}

double VectorQuantile(vector<double>& dat,double p)
{
    sort(dat.begin(),dat.end());
    double Q_idx=1+(dat.size()-1)*p;
    int Q_idx_integer=(int)Q_idx;
    double Q_idx_decimal=Q_idx-Q_idx_integer;
    double Q=dat[Q_idx_integer-1]+(dat[Q_idx_integer]-dat[Q_idx_integer-1])*Q_idx_decimal;    
    return Q;
}

int VectorIQR(vector<double>& dat)
{
    double IQR=VectorQuantile(dat,0.75)-VectorQuantile(dat,0.25);
    double thresh=VectorQuantile(dat,0.75)+10*IQR;
    for(int i=0;i<dat.size();i++)
    {
        if(dat[i]>thresh){
            return 1;
        }
    }
    return 0;
}

double VectorSum(vector<double>& dat)
{
    double sum=0;
    for(int i=0;i<dat.size();i++)
    {
        sum+=dat[i];
    }
    return sum;
}