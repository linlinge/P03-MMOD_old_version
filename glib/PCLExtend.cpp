#include "PCLExtend.h"
double ComputeMeanDistance(const pcl::PointCloud<PointType>::ConstPtr cloud)
{
	double res = 0.0;
	int n_points = 0;
	int nres;
	std::vector<int> indices(2);
	std::vector<float> sqr_distances(2);
	pcl::search::KdTree<PointType> tree;
	tree.setInputCloud(cloud);

	for (size_t i = 0; i < cloud->size(); ++i)
	{
		if (!std::isfinite((*cloud)[i].x))
		{
			continue;
		}
		nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
		if (nres == 2)
		{
			res += sqrt(sqr_distances[1]);
			++n_points;
		}
	}
	if (n_points != 0)
	{
		res /= n_points;
	}
	return res;
}


double ComputeMaxDistance(const pcl::PointCloud<PointType>::ConstPtr cloud)
{
	double rst = 0.0;
	int nres;
	std::vector<int> indices(2);
	std::vector<float> sqr_distances(2);
	pcl::search::KdTree<PointType> tree;
	tree.setInputCloud(cloud);

	for (size_t i = 0; i < cloud->size(); ++i)
	{
		if (!std::isfinite((*cloud)[i].x))
		{
			continue;
		}
		nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
		if (nres == 2)
		{
			double dist_tmp=sqrt(sqr_distances[1]);
			rst = rst>dist_tmp ? rst:dist_tmp;
		}
	}

	return rst;
}

// compute centroid of point cloud
PointType ComputeCentroid(const pcl::PointCloud<PointType>::ConstPtr cloud)
{
   PointType centroid;
   centroid.x=0;
   centroid.y=0;
   centroid.z=0;
   for(int i=0;i<cloud->points.size();i++){
	   centroid.x+=cloud->points[i].x;
	   centroid.y+=cloud->points[i].y;
	   centroid.z+=cloud->points[i].z;
   }
   centroid.x=centroid.x*1.0/cloud->points.size();
   centroid.y=centroid.y*1.0/cloud->points.size();
   centroid.z=centroid.z*1.0/cloud->points.size();
   return centroid;
}

vector<double> StatisticNearestDistance(const pcl::PointCloud<PointType>::ConstPtr cloud)
{
	int n_points = 0;
	int nres;
	std::vector<int> indices(2);
	std::vector<float> sqr_distances(2);
	vector<double> rst;
	pcl::search::KdTree<PointType> tree;
	tree.setInputCloud(cloud);

	for (size_t i = 0; i < cloud->size(); ++i)
	{ 
		if (!std::isfinite((*cloud)[i].x))
		{
			continue;
		}
		nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
		if (nres == 2)
		{			
			rst.push_back(sqr_distances[1]);
			++n_points;
		}
	}	
	
	return rst;
}

void TransformPointCloud(pcl::PointCloud<PointType>::Ptr cloud, pcl::PointCloud<PointType>::Ptr cloud_tf,Eigen::Affine3f tf)
{
	cloud_tf->points.resize(cloud->points.size());	
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++)
	{
		Eigen::Vector3f v1(cloud->points[i].x,cloud->points[i].y,cloud->points[i].z);		
		Eigen::Vector3f v2=tf*v1;
		cloud_tf->points[i].x=v2(0,0);
		cloud_tf->points[i].y=v2(1,0); 
		cloud_tf->points[i].z=v2(2,0);
	}
}

void EvalAndEvec::GetEvalAndEvec(pcl::PointCloud<PointType>::Ptr cloud)
{
	Eigen::Vector4f centroid;
	Eigen::Matrix3f covariance;
	pcl::compute3DCentroid(*cloud, centroid);
	pcl::computeCovarianceMatrixNormalized(*cloud, centroid, covariance);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eig_vec = eigen_solver.eigenvectors();
	Eigen::Vector3f eig_val = eigen_solver.eigenvalues();
	
	// eigenvalue
	eigenvalue_[0]=eig_val(0);
	eigenvalue_[1]=eig_val(1);
	eigenvalue_[2]=eig_val(2);

	// eigenvector
	eig_vec.col(2) = eig_vec.col(0).cross(eig_vec.col(1));
	eig_vec.col(0) = eig_vec.col(1).cross(eig_vec.col(2));
	eig_vec.col(1) = eig_vec.col(2).cross(eig_vec.col(0));
	eigenvector_[0].x=eig_vec(0,0);eigenvector_[0].y=eig_vec(1,0);eigenvector_[0].z=eig_vec(2,0);
	eigenvector_[1].x=eig_vec(0,1);eigenvector_[1].y=eig_vec(1,1);eigenvector_[1].z=eig_vec(2,1);
	eigenvector_[2].x=eig_vec(0,2);eigenvector_[2].y=eig_vec(1,2);eigenvector_[2].z=eig_vec(2,2);
}