#pragma once

#include <vector>
#include <iostream>
#include <cstring>
#include <fstream>

#include "Eigen.h"
#include "FreeImageHelper.h"
#include "Timer.h"

using namespace std;

struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// color stored as 4 unsigned char
	Vector4uc color;
};

typedef unsigned char BYTE;

//implement the CameraIntrinsic
struct CameraIntrinsic
{
	Eigen::Matrix3f Intrinsic;
	float focal_length;
	float camera_width;
	float camera_height;
	float image_width;;
	float image_height;
	float center_width;
	float center_height;
	float scale;

	CameraIntrinsic(float focal_length=1, float camera_width=1, float camera_height=1,
		float image_width=1, float image_height=1, float center_width=1, float center_height=1, float scale=1):
		focal_length(focal_length), camera_height(camera_height), camera_width(camera_width),
		image_height(image_height), image_width(image_width), center_height(center_height), center_width(center_width),
		scale(scale)

	{
		Intrinsic << focal_length * image_width / camera_width, 0.0, center_width,
			0.0, focal_length* image_height / camera_height, center_height,
			0.0, 0.0, 1.0;
	}
	CameraIntrinsic(Eigen::Matrix3f in) {
		Intrinsic = in;
		image_width = 640;
		image_height = 480;
		scale = 5000.0f;
	}

	Eigen::Matrix3f get_Intrinsic()
	{
		return Intrinsic;
	}

};

struct FrameData { 
	cv::Mat image;

	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;

	Vertex* vertices;
	Eigen::Matrix4f extrinsics = Eigen::Matrix4f::Identity();

	size_t frameId;

	std::vector<Eigen::Matrix4f> potentialExtrinsics;
	
	std::vector<double> costs;

	/// <summary>
	/// set current extrinsic to the weighted average of all potential extrinsics
	/// weights are calculated by their respective cost
	/// weights are inverted because the lower the cost the higher its significance
	/// </summary>
	void UpdateExtrinsics() {
		size_t size = costs.size();

		if (size == 0) {
			extrinsics = Eigen::Matrix4f::Identity();
			return;
		}
		if (size == 1) {
			extrinsics = potentialExtrinsics[0];
			return;
		}

		extrinsics = Eigen::Matrix4f::Zero();
		double totalWeight = 0;
		for (size_t i = 0; i < size; i++)
		{
			double weight = 1.0 / costs[i];
			totalWeight += weight;
			extrinsics += weight * potentialExtrinsics[i];
		}

		extrinsics /= totalWeight;
	}
};

// reads sensor files according to https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
class VirtualSensor
{
public:
	//m_maxIdx limits the maximum number of frames that are processed
	VirtualSensor(CameraIntrinsic x ): m_currentIdx(-1), FRAME_INCREMENT(1), camera(x), m_maxIdx(50)
	{
		m_colorImageWidth = camera.image_width;
		m_colorImageHeight = camera.image_height;
		m_depthImageWidth = camera.image_width;
		m_depthImageHeight = camera.image_height;
		// intrinsics
		m_colorIntrinsics = camera.Intrinsic;
		m_depthIntrinsics = m_colorIntrinsics;
	}

	~VirtualSensor()
	{
		SAFE_DELETE_ARRAY(m_depthFrame);
		SAFE_DELETE_ARRAY(m_colorFrame);
	}

	bool Init(const std::string& datasetDir)
	{
		m_baseDir = datasetDir;

		// read filename lists
		if (!ReadFileList(datasetDir + "depth.txt", m_filenameDepthImages, m_depthImagesTimeStamps)) return false;
		if (!ReadFileList(datasetDir + "rgb.txt", m_filenameColorImages, m_colorImagesTimeStamps)) return false;

		// read tracking
		if (!ReadTrajectoryFile(datasetDir + "groundtruth.txt", m_trajectory, m_trajectoryTimeStamps)) return false;

		if (m_filenameDepthImages.size() != m_filenameColorImages.size()) return false;

		//m_colorExtrinsics.setIdentity();
		//m_depthExtrinsics.setIdentity();

		m_depthFrame = new float[m_depthImageWidth*m_depthImageHeight];
		for (unsigned int i = 0; i < m_depthImageWidth*m_depthImageHeight; ++i) m_depthFrame[i] = 0.5f;

		m_colorFrame = new BYTE[4* m_colorImageWidth*m_colorImageHeight];
		for (unsigned int i = 0; i < 4*m_colorImageWidth*m_colorImageHeight; ++i) m_colorFrame[i] = 255;


		m_currentIdx = -1;
		return true;
	}

	unsigned int GetCurrentIDX() {
		return m_currentIdx;
	}

	bool ProcessNextFrame()
	{
		if (m_currentIdx == -1)	m_currentIdx = 0;
		else m_currentIdx += FRAME_INCREMENT;

		if ((unsigned int)m_currentIdx >= (unsigned int)m_maxIdx) return false;

		std::cout << "ProcessNextFrame [" << m_currentIdx << " | " << m_filenameColorImages.size() << "]" << std::endl;

		m_frameData.frameId = m_currentIdx;

		FreeImageB rgbImage;
		rgbImage.LoadImageFromFile(m_baseDir + m_filenameColorImages[m_currentIdx]);
		memcpy(m_colorFrame, rgbImage.data, 4 * 640 * 480);
		m_frameData.image = rgbImage.mat;

		// get the keypoints and descriptors of current frame
		auto detector = cv::SIFT::create();
		detector->detectAndCompute(rgbImage.mat, cv::noArray(), m_frameData.keypoints, m_frameData.descriptors);

		// depth images are scaled by 5000 (see https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)
		FreeImageU16F dImage;
		dImage.LoadImageFromFile(m_baseDir + m_filenameDepthImages[m_currentIdx]);

		for (unsigned int i = 0; i < m_depthImageWidth*m_depthImageHeight; ++i)
		{
			if (dImage.data[i] == 0)
				m_depthFrame[i] = MINF;
			else
				m_depthFrame[i] = dImage.data[i] * 1.0f / (float)camera.scale;
		}

		// generate vertices with initial extrinsics
		float* depthMap = GetDepth();
		Matrix3f depthIntrinsics = GetDepthIntrinsics();
		Matrix3f depthIntrinsicsInv = depthIntrinsics.inverse();
		BYTE* colorMap = GetColorRGBX();
		unsigned int pixelW = GetDepthImageWidth();
		unsigned int pixelH = GetDepthImageHeight();
		m_frameData.vertices = new Vertex[pixelH*pixelW];
		Vertex tempV = Vertex();
		//StartCounter();
#pragma omp parallel for // first iteration takes double time than without parallel, after it takes a quarter time
		for (int y = 0; y < pixelH; y++) {
			for (unsigned int x = 0; x < pixelW; x++) {
				unsigned int idx = x + pixelW * y;//the current idx value
				if (depthMap[idx] == MINF) { //if the depth is not available,then define the vertices as suggested in the TODO
					m_frameData.vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
					m_frameData.vertices[idx].color = Vector4uc(0, 0, 0, 0);

					continue;
				}
				// initiate color vector:
				m_frameData.vertices[idx].color = Vector4uc(colorMap[idx * 4], colorMap[idx * 4 + 1], colorMap[idx * 4 + 2], colorMap[idx * 4 + 3]);
				// compute the position vector:
				float depth = depthMap[idx];
				// back-projection to camera space:
				Vector3f positionI = Vector3f(x * depth, y * depth, depth);
				Vector3f positionC = depthIntrinsicsInv * positionI;
				Vector4f positionC2 = Vector4f(positionC(0, 0), positionC(1, 0), positionC(2, 0), 1);
				m_frameData.vertices[idx].position = positionC2;
			}
		}
		//cout << "Get frame data duration (ms): " << GetCounter() << endl;

		return true;
	}

	unsigned int GetCurrentFrameCnt()
	{
		return (unsigned int)m_currentIdx;
	}

	// get current color data
	BYTE* GetColorRGBX()
	{
		return m_colorFrame;
	}
	// get current depth data
	float* GetDepth()
	{
		return m_depthFrame;
	}

	// color camera info
	Eigen::Matrix3f GetColorIntrinsics()
	{
		return m_colorIntrinsics;
	}
	

	unsigned int GetColorImageWidth()
	{
		return m_colorImageWidth;
	}

	unsigned int GetColorImageHeight()
	{
		return m_colorImageHeight;
	}

	size_t GetTotalColorPixelAmount() {
		return (size_t)GetColorImageWidth() * (size_t)GetColorImageHeight();
	}

	// depth (ir) camera info
	Eigen::Matrix3f GetDepthIntrinsics()
	{
		return m_depthIntrinsics;
	}
	

	unsigned int GetDepthImageWidth()
	{
		return m_colorImageWidth;
	}

	unsigned int GetDepthImageHeight()
	{
		return m_colorImageHeight;
	}

	Eigen::Matrix4f GetGroundTruth(size_t idx) {
		Eigen::Matrix4f f;
		double timestamp = m_depthImagesTimeStamps[idx];
		double min = std::numeric_limits<double>::max();
		int k = 0;
		for (unsigned int i = 0; i < m_trajectory.size(); ++i)
		{
			double d = fabs(m_trajectoryTimeStamps[i] - timestamp);
			if (min > d)
			{
				min = d;
				k = i;
			}
		}
		
		
		
		return m_trajectory[k];
		
	}

private:

	bool ReadFileList(const std::string& filename, std::vector<std::string>& result, std::vector<double>& timestamps)
	{
		std::ifstream fileDepthList(filename, std::ios::in);
		if (!fileDepthList.is_open()) return false;
		result.clear();
		timestamps.clear();
		std::string dump;
		std::getline(fileDepthList, dump);
		std::getline(fileDepthList, dump);
		std::getline(fileDepthList, dump);
		while (fileDepthList.good())
		{
			double timestamp;
			fileDepthList >> timestamp;
			std::string filename;
			fileDepthList >> filename;
			if (filename == "") break;
			timestamps.push_back(timestamp);
			result.push_back(filename);
		}
		fileDepthList.close();
		return true;
	}

	bool ReadTrajectoryFile(const std::string& filename, std::vector<Eigen::Matrix4f>& result, std::vector<double>& timestamps)
	{
		std::ifstream file(filename, std::ios::in);
		if (!file.is_open()) return false;
		result.clear();
		std::string dump;
		std::getline(file, dump);
		std::getline(file, dump);
		std::getline(file, dump);

		while (file.good())
		{
			double timestamp;
			file >> timestamp;
			Eigen::Vector3f translation;
			file >> translation.x() >> translation.y() >> translation.z();
			Eigen::Quaternionf rot;
			file >> rot;

			Eigen::Matrix4f transf;
			transf.setIdentity();
			transf.block<3, 3>(0, 0) = rot.toRotationMatrix();
			transf.block<3, 1>(0, 3) = translation;

			if (rot.norm() == 0) break;

			//transf = transf.inverse().eval();

			timestamps.push_back(timestamp);
			result.push_back(transf);
		}
		file.close();
		return true;
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
	// current frame index
	int m_currentIdx;
	int m_maxIdx;
	const int FRAME_INCREMENT;

	// frame data
	float* m_depthFrame;
	BYTE* m_colorFrame;
	Eigen::Matrix4f m_currentTrajectory;

	// color camera info
	Eigen::Matrix3f m_colorIntrinsics;
	//Eigen::Matrix4f m_colorExtrinsics;
	unsigned int m_colorImageWidth;
	unsigned int m_colorImageHeight;

	// depth (ir) camera info
	Eigen::Matrix3f m_depthIntrinsics;
	//Eigen::Matrix4f m_depthExtrinsics;
	unsigned int m_depthImageWidth;
	unsigned int m_depthImageHeight;

	// base dir
	std::string m_baseDir;
	// filenamelist depth
	std::vector<std::string> m_filenameDepthImages;
	std::vector<double> m_depthImagesTimeStamps;
	// filenamelist color
	std::vector<std::string> m_filenameColorImages;
	std::vector<double> m_colorImagesTimeStamps;
	// trajectory(for evaluation)
	std::vector<Eigen::Matrix4f> m_trajectory;
	std::vector<double> m_trajectoryTimeStamps;
	CameraIntrinsic camera;

	FrameData m_frameData;
}; 



