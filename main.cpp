#include <iostream>
#include <fstream>
#include <array>

#include "opencv2/features2d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include <opencv2/imgproc.hpp>

#include "Eigen.h"
#include "StreamManage.h"

using namespace std;




struct ImagePairFeaturePoints
{

	ImagePairFeaturePoints() {
		m_GoodMatches.reserve(10);
	};

	size_t m_iImgID1 = 0;
	size_t m_iImgID2 = 0;

	vector<DMatch> m_GoodMatches;
};

void LoadAllImagePairFeatures(ImagePairFeaturePoints* a_pFeaturePairCollection, const vector<string>& a_ImagePaths)
{
	size_t l_iImageCount = a_ImagePaths.size();
	size_t l_iPairIndex = 0;
	for (size_t l_iImageIndex = 0; l_iImageIndex < l_iImageCount -1; l_iImageIndex++)
	{
		for (size_t l_iSecondImageInPairIndex = l_iImageIndex + 1; l_iSecondImageInPairIndex < l_iImageCount; l_iSecondImageInPairIndex++, l_iPairIndex++)
		{
			ImagePairFeaturePoints& l_refCurrenPair = a_pFeaturePairCollection[l_iPairIndex];
			l_refCurrenPair.m_iImgID1 = l_iImageIndex;
			l_refCurrenPair.m_iImgID2 = l_iSecondImageInPairIndex;

			auto img1 = imread(a_ImagePaths[l_iImageIndex]);
			auto img2 = imread(a_ImagePaths[l_iSecondImageInPairIndex]);
			// detect keypoints
			auto detector = SIFT::create();
			vector<KeyPoint> keypoints1, keypoints2;
			Mat descriptors1, descriptors2;
			detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
			detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
			// match keypoints
			auto matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
			vector<vector<DMatch>> knn_matches;
			matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
			// filter matches
			const auto ratio_thresh = 0.2f;
			for (size_t i = 0; i < knn_matches.size(); i++)
			{
				if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
					l_refCurrenPair.m_GoodMatches.push_back(knn_matches[i][0]);
				}
			}

			std::cout << "storing pair : (" << l_iImageIndex << " , " << l_iSecondImageInPairIndex << ")\n";
		}
	}
	
}

void printMatch(DMatch a) {
	cout << a.imgIdx <<"# "<< a.queryIdx << "# " << a.trainIdx << "# " << a.distance << endl;
	

}


int main() {
	
	
	
	Eigen::Matrix3f intrinsic;
	intrinsic << 517.306408, 0.000000, 318.643040, 
		0.000000, 516.469215, 255.313989, 
		0.0, 0.0, 1.0;
	CameraIntrinsic camera(intrinsic);


	
	
	camera.image_width = 640;
	camera.image_height = 480;

	const std::string fileIn = "../../Data/rgbd_dataset_freiburg1_xyz/";
	const std::string fileOut = "output/";

	RBG_DStream manager = RBG_DStream(fileIn,fileOut,camera);
	manager.Init();
	while(manager.ProcessNextFrame()) {}

	manager.printPose();

	cout << "Finish\n";
	
	manager.end();



	return 1;

	

	
	/*
	//declaring the parameters of the camera
	float focal_length;
	float camera_width;
	float camera_height;
	float image_width;;
	float image_height;
	float center_width;
	float center_height;
	float scale;

	//generate a camera(camera intrinsics will be caculated automatically)
	CameraIntrinsic camera(focal_length , camera_width, camera_height,
		image_width , image_height , center_width, center_height, scale);
	
	//setting up file path to pictures
	const std::string filePath = "";

	//setting up the stream manager
	RBG_DStream streamManager(filePath, camera);

	//init the stream manager(extracting all files from the filePath)
	streamManager.Init();
	//processing the features of the pictures:
	streamManager.SIFT();
	//caculate the pose of each frame basing on the loss function(implemented within the streamManager):
	streamManager.calPoses();
	//merge the pictures together and output a .off point cloud file.
	streamManager.PrintPointCloud();
	//end the stream manager
	streamManager.end();*/






}
