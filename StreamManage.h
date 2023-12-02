#pragma once
#include "Eigen.h"
#include "opencv2/features2d.hpp"
#include "VirtualSensor.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "Optimizer.h"
#include <iomanip>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "GridGenerator.h"
#include <omp.h>

using namespace cv;
using namespace std;



class Chunk
{
	unsigned int N;
public:


};

class Correspondences {
public:
	Vector3f source;
	Vector3f target;

	Correspondences(Vector3f s, Vector3f t) : source(s), target(t)
	{}
};

enum class PrintType {
	PrintPointCloud,
	PrintOccupancyGrid,
	PrintOctree
};

class RBG_DStream
{
	Chunk localM;
	VirtualSensor sensor;
	std::string fileNameIn;
	std::string fileNameOut;
	CameraIntrinsic camera;

	int windowSize = 8;
	int totalRefSize = 16;

	int m_iNumOfThreads = 0;

	bool printLastOnly = true;
	PrintType printType = PrintType::PrintPointCloud;
	OccupancyGrid* occupancyGrid = nullptr;
	Octree* octree = nullptr;
	vector<vector<Correspondences>> m_vecCorrPerThread;

public:
	std::vector<FrameData> frames;

	Ptr<SIFT> featureDetector;
	Ptr<DescriptorMatcher> matcher;
	std::vector<KeyPoint> keyPoints;

public:

	RBG_DStream(const std::string& fileNameIn, const std::string& fileNameOut, CameraIntrinsic c) :
		fileNameIn(fileNameIn), fileNameOut(fileNameOut), camera(c), sensor(VirtualSensor(c)), m_iNumOfThreads(omp_get_max_threads()), m_vecCorrPerThread(windowSize > m_iNumOfThreads ? m_iNumOfThreads : windowSize)
	{
		//sensor = VirtualSensor(camera);
		featureDetector = SIFT::create();
		matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		keyPoints = std::vector<KeyPoint>();
		//vertices = new Vertex[480*640];
		std::cout << "NumOfThreads: " << m_iNumOfThreads<<"\n";
	}

	~RBG_DStream() {
		if (occupancyGrid != nullptr)
			delete occupancyGrid;
		if (octree != nullptr)
			delete octree;
	}

	Matrix4f PoseToMatrix(double* pose) {
		double* rotation = pose;
		double rotationMatrix[9];
		ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);
		// Create the 4x4 transformation matrix.
		Matrix4f matrix;
		matrix.setIdentity();
		double* translation = pose + 3;
		matrix(0, 0) = float(rotationMatrix[0]);	matrix(0, 1) = float(rotationMatrix[3]);	matrix(0, 2) = float(rotationMatrix[6]);	matrix(0, 3) = float(translation[0]);
		matrix(1, 0) = float(rotationMatrix[1]);	matrix(1, 1) = float(rotationMatrix[4]);	matrix(1, 2) = float(rotationMatrix[7]);	matrix(1, 3) = float(translation[1]);
		matrix(2, 0) = float(rotationMatrix[2]);	matrix(2, 1) = float(rotationMatrix[5]);	matrix(2, 2) = float(rotationMatrix[8]);	matrix(2, 3) = float(translation[2]);

		return matrix;
	}

	float* MatrixToPose(Matrix4f m) {
		float* pose = new float[6];

		// rotation matrix to euler angles https://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-rotation-matrix
		float alpha = atan2(m(2, 1), m(2, 2));
		float beta = atan2(-m(2, 0), sqrt(pow(m(2, 1), 2) + pow(m(2, 2), 2)));
		float gamma = atan2(m(1, 0), m(0, 0));

		float x = m(0, 3);
		float y = m(1, 3);
		float z = m(2, 3);

		pose[0] = alpha;
		pose[1] = beta;
		pose[2] = gamma;
		pose[3] = x;
		pose[4] = y;
		pose[5] = z;

		return pose;
	}

	bool Init() {
		if (!sensor.Init(fileNameIn))
		{
			std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
			return false;
		}

		if (printType != PrintType::PrintPointCloud) {
			Vector3f minCoord;
			minCoord << -2.77f, -1.77f, -0.05f;
			Vector3f maxCoord;
			maxCoord << 1.02f, 2.61f, 1.83f;
			switch (printType) {
			case PrintType::PrintOccupancyGrid:
				occupancyGrid = new OccupancyGrid(512, minCoord, maxCoord);
				break;
			case PrintType::PrintOctree:
				octree = new Octree(512, minCoord, maxCoord);
				break;
			}
		}

		omp_set_num_threads(m_vecCorrPerThread.size());
	}

	bool ProcessNextFrame() {
		if (!sensor.ProcessNextFrame()) {
			return false;
		}
		// store the new frame data for later matching
		FrameData frameData = sensor.m_frameData;

		//vector frames can be reused as a states-observer

		frames.push_back(frameData);
		FrameData& currentFrame = frames[frames.size() - 1];

		// get image width for vertex index calculation
		int width = sensor.GetColorImageWidth();

		StartCounter();

		if (currentFrame.frameId == 0) {
			currentFrame.extrinsics = sensor.GetGroundTruth(frames[0].frameId);
		}
		else {
			currentFrame.extrinsics = frames[frames.size() - 2].extrinsics;
		}

		// setup the window to work on
		totalRefSize = max(windowSize, totalRefSize);
		int windowStart = max(0, (int)currentFrame.frameId - windowSize);
		int windowEnd = (int)currentFrame.frameId;
		int refStart = max(0, (int)currentFrame.frameId - totalRefSize);
		for (int sourceId = windowEnd; sourceId >= windowStart; sourceId -= sensor.FRAME_INCREMENT)
		{
			FrameData& sourceFrame = frames[sourceId / sensor.FRAME_INCREMENT];
			vector<Correspondences> corr;

			#pragma omp parallel for schedule(guided) default(shared)
			for (int targetId = refStart; targetId <= windowEnd; targetId += sensor.FRAME_INCREMENT)
			{
				if (targetId == sourceId) continue;

				FrameData targetFrame = frames[targetId / sensor.FRAME_INCREMENT];

				// match features
				Mat descriptors = targetFrame.descriptors;
				auto matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
				vector<vector<DMatch>> knn_matches;
				matcher->knnMatch(sourceFrame.descriptors, descriptors, knn_matches, 2);
				// filter matches
				vector<DMatch> goodMatches;
				const auto ratio_thresh = 0.2f;
				for (size_t i = 0; i < knn_matches.size(); i++)
				{
					if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
						goodMatches.push_back(knn_matches[i][0]);
					}
				}

				for (size_t j = 0; j < goodMatches.size(); j++)
				{
					// get keypoint IDs from match
					int targetKeyPointID = goodMatches[j].trainIdx;
					int sourceKeyPointID = goodMatches[j].queryIdx;
					// get pixel coordinates from keypoints
					Point2f sourcePixelCoord = sourceFrame.keypoints[sourceKeyPointID].pt;
					Point2f targetPixelCoord = targetFrame.keypoints[targetKeyPointID].pt;

					// get world coordinates from pixel coordinates
					Vector4f sourcePoint = sourceFrame.vertices[int(sourcePixelCoord.x) + width * int(sourcePixelCoord.y)].position;
					Vector4f targetPoint = targetFrame.vertices[int(targetPixelCoord.x) + width * int(targetPixelCoord.y)].position;

					// don't use the keypoints if one of them don't have a valid 3D vector
					if (sourcePoint.x() == MINF || targetPoint.x() == MINF) continue;

					// also use their current extrinsics
					//sourcePoint = sourceFrame.extrinsics * sourcePoint;
					targetPoint = targetFrame.extrinsics * targetPoint;

					// only use x,y,z for optimization
					m_vecCorrPerThread[omp_get_thread_num()].push_back(Correspondences(sourcePoint.head<3>(), targetPoint.head<3>()));
				}
			}

			for (int l_iVecIndex = 0; l_iVecIndex < m_vecCorrPerThread.size(); l_iVecIndex++)
			{
				corr.insert(corr.end(), m_vecCorrPerThread[l_iVecIndex].begin(), m_vecCorrPerThread[l_iVecIndex].end());
				m_vecCorrPerThread[l_iVecIndex].clear();
			}

			if (corr.size() > 2) {
				// optimize the transformation (alpha, beta, gamma, x, y, z)
				double pose[6] = { 0,0,0,0,0,0 };
				vector<Vector3f> sourcePoints;
				vector<Vector3f> targetPoints;
				for (size_t i = 0; i < corr.size(); i++)
				{
					sourcePoints.push_back(corr[i].source);
					targetPoints.push_back(corr[i].target);
				}

				// optimize with loss function
				Optimizer optimizer;
				optimizer.estimatePose(sourcePoints, targetPoints, pose);

				// store transformation as extrinsics
				// Convert the rotation from SO3 to matrix notation (with column-major storage).0
				Matrix4f matrix = PoseToMatrix(pose);
				// store matrix
				// only use extrinsic if it was optimized
				if (optimizer.terminationType == ceres::TerminationType::CONVERGENCE
					&& optimizer.finalCost > 0) {
					sourceFrame.extrinsics = matrix;
				}
				else {
					cout << "extrinsics not updated !!!\n";
				}
			}
			else {
				cout << "extrinsics not updated !!!\n";
			}
		}

		cout << "Optimization duration at frame " << frameData.frameId << " (ms) : " << GetCounter() << endl;

		StartCounter();

		switch (printType) {
		case PrintType::PrintPointCloud:
			if (!printLastOnly || currentFrame.frameId + sensor.FRAME_INCREMENT >= sensor.m_maxIdx)
				PrintPointCloud();
			break;
		case PrintType::PrintOccupancyGrid:
			occupancyGrid->InsertPoints(currentFrame.vertices, currentFrame.extrinsics, sensor.GetTotalColorPixelAmount());
			if (!printLastOnly || currentFrame.frameId + sensor.FRAME_INCREMENT >= sensor.m_maxIdx)
				PrintOccupancyGrid();
			break;
		case PrintType::PrintOctree:
			octree->InsertPoints(currentFrame.vertices, currentFrame.extrinsics, sensor.GetTotalColorPixelAmount());
			if (!printLastOnly || currentFrame.frameId + sensor.FRAME_INCREMENT >= sensor.m_maxIdx)
				PrintOctree();
			break;
		}


		cout << "Print duration (ms): " << GetCounter() << endl;

		return true;
	}

private:
	bool valid_point(Vertex* vertices, unsigned int idx) {
		if (vertices[idx].position == Vector4f(MINF, MINF, MINF, MINF)) return false;

		return true;

	}

	//void ShowImage(vector<Point> points, Mat img)
	//{
	//	line(img , Point(0,0), Point(0,100), Scalar(0,255,0), 2, LINE_8);
	//	
	//	imshow("Output", img);
	//	
	//}

	bool WriteMesh(unsigned int width, unsigned int height, const std::string& filename)
	{
		unsigned int nVertices = 0;
		for (int x = 0; x < frames.size(); x++) {
			for (size_t i = 0; i < width * height; i++)
			{
				if (valid_point(frames[x].vertices, i)) {
					++nVertices;
				}
			}
		}


		unsigned nFaces = 0;
		std::ofstream outFile;
		outFile.open(filename, std::ios_base::trunc);
		if (!outFile.is_open()) return false;


		outFile << "COFF" << std::endl;

		outFile << "# numVertices numFaces numEdges" << std::endl;

		outFile << nVertices << " " << nFaces << " 0" << std::endl;

		// TODO: save vertices
		for (unsigned int k = 0; k < frames.size(); k++) {
			for (unsigned int y = 0; y < height; y++) {
				for (unsigned int x = 0; x < width; x++) {
					unsigned int idx = x + y * width;
					if (!valid_point(frames[k].vertices, idx)) {
						//outFile << 0.0f << " " << 0.0f << " " << 0.0f;
						//outFile << "	" << "255 " << "255 " << "255 " << 255 << std::endl;
						continue;
					}
					//Vector4f transformedPoint = sensor.GetGroundTruth(frames[k].frameId) * frames[k].vertices[idx].position;
					Vector4f transformedPoint = frames[k].extrinsics * frames[k].vertices[idx].position;
					outFile << transformedPoint(0, 0) << " " << transformedPoint(1, 0)
						<< " " << transformedPoint(2, 0) << " ";
					outFile << frames[k].vertices[idx].color(0, 0) * 1.0f << " " << frames[k].vertices[idx].color(1, 0) * 1.0f << " "
						<< frames[k].vertices[idx].color(2, 0) * 1.0f << " " << frames[k].vertices[idx].color(3, 0) * 1.0f << std::endl;
				}
			}


		}


		outFile.close();

		return true;
	}

	bool WritePose(const std::string& filename) {

		std::ofstream outFile;
		outFile.open(filename, std::ios_base::trunc);
		if (!outFile.is_open()) return false;
		

		// just checking
		if (frames.size() == 0) {
			outFile.close();
			return true;
		}
		
		for (int i = 0; i < frames.size(); i++) {

			float* ePose = MatrixToPose(frames[i].extrinsics);
			float* tPose = MatrixToPose(sensor.GetGroundTruth(frames[i].frameId));

			outFile << tPose[0] << ' ' << tPose[1] << ' ' << tPose[2] << ' ' << tPose[3] << ' ' << tPose[4] << ' ' << tPose[5]<<' '
				<<ePose[0] << ' ' << ePose[1] << ' ' << ePose[2] << ' ' << ePose[3] << ' ' << ePose[4] << ' ' << ePose[5] << endl;
		}
		return true;
	}
public:
	bool PrintPointCloud() {
		std::stringstream ss;
		ss << fileNameOut << sensor.m_frameData.frameId << ".off";
		if (!WriteMesh(sensor.GetDepthImageWidth(), sensor.GetDepthImageHeight(), ss.str()))
		{
			std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
			return false;

		}

		return true;
	}

	bool PrintOccupancyGrid() {
		size_t size = sensor.GetTotalColorPixelAmount();
		std::stringstream ss;
		ss << fileNameOut << sensor.m_frameData.frameId << ".off";
		return occupancyGrid->WriteMesh(ss.str(), size);
	}

	bool PrintOctree() {
		std::stringstream ss;
		ss << fileNameOut << sensor.m_frameData.frameId << ".off";
		return octree->WriteMesh(ss.str());
	}

	void EvaluateDifferenceWithGroundTruth(std::string a_strFileName)
	{
		std::ifstream file(a_strFileName, std::ios::in);
		if (!file.is_open()) return;
		std::string dump;
		std::getline(file, dump);
		std::getline(file, dump);
		std::getline(file, dump);

		int l_iFrameIndex = 0;
		int l_iLineIndex = -1;
		Eigen::Vector3f l_v3GroundTruthTranslationBase;
		Eigen::Vector3f l_v3EvaluatedTranslationBase = frames[0].extrinsics.block<3, 1>(0, 3);
		while (file.good()) {
			if (l_iFrameIndex >= (frames.size())) { return; }

			l_iLineIndex++;
			FrameData& l_CurrentFrameData = frames[l_iFrameIndex];
			const int l_iCurrentFrameID = l_CurrentFrameData.frameId;

			if (l_iLineIndex != l_iCurrentFrameID)
			{
				file.ignore(numeric_limits<streamsize>::max(), file.widen('\n'));
				continue;
			}

			double timestamp;
			file >> timestamp;

			Eigen::Vector3f l_v3EvaluatedTranslation = l_CurrentFrameData.extrinsics.block<3, 1>(0, 3);
			Eigen::Quaternionf l_qRotation(l_CurrentFrameData.extrinsics.block<3, 3>(0, 0));

			Eigen::Vector3f l_v3GroundTruthTranslation;

			file >> l_v3GroundTruthTranslation.x() >> l_v3GroundTruthTranslation.y() >> l_v3GroundTruthTranslation.z();
			//find the base of the translation so that the trajectory starts from (0,0,0)
			if (l_iLineIndex == 0) {
				l_v3GroundTruthTranslationBase = l_v3GroundTruthTranslation;

			}
			l_v3GroundTruthTranslation -= l_v3GroundTruthTranslationBase;
			Eigen::Quaternionf l_qGroundTruthRotation;
			file >> l_qGroundTruthRotation;

			file.ignore(numeric_limits<streamsize>::max(), file.widen('\n'));

			std::cout << "\n--- Timestamp : " << timestamp << "  FrameID: " << l_iCurrentFrameID << "--- \n"
				<< std::setw(15) << "GroundTruth: " << std::setw(15) << l_v3GroundTruthTranslation.x() << std::setw(15) << l_v3GroundTruthTranslation.y() << std::setw(15) << l_v3GroundTruthTranslation.z() << std::setw(15) << l_qGroundTruthRotation.x() << std::setw(15) << l_qGroundTruthRotation.y() << std::setw(15) << setfill(' ') << l_qGroundTruthRotation.z() << std::setw(15) << l_qGroundTruthRotation.w() << "\n"
				<< std::setw(15) << "Evaluated: " << std::setw(15) << l_v3EvaluatedTranslation.x() << std::setw(15) << l_v3EvaluatedTranslation.y() << std::setw(15) << l_v3EvaluatedTranslation.z() << std::setw(15) << l_qRotation.x() << std::setw(15) << l_qRotation.y() << std::setw(15) << l_qRotation.z() << std::setw(15) << l_qRotation.w() << "\n";

			l_iFrameIndex++;
		}
	}

	bool printPose() {
		std::stringstream ss;
		ss << fileNameOut << "EstimatedPoses" << ".txt";
		if (!WritePose(ss.str()))
		{
			std::cout << "Failed to write Pose!\nCheck file path!" << std::endl;
			return false;

		}

		return true;

	}


	void end()
	{
	}



};


