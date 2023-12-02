# **Project structure**

-->Data
	-->datasets
-->Lib
	-->OpenCV
-->Ceres
	-->Eigen
-->Flann-1.8.4
-->FreeImage-3.18.0
-->Glog
 -->Final_Project
	-->Build
		-->output
	-->main.cpp
	-->CMakeLists.txt


## **Dependencies/Libraries used**
OpenCV
Ceres
Eigen
Flann-1.8.4
FreeImage-3.18.0
Glog
If you are having problems with OpenCV please check the record on Day 24/JUNE/2022 and 26/JUNE/2022 in the Implementation_Diary.txt.
Implementation_Diary.txt keeps track of what each member is contributing to this project and our workflow.

## **To build the project**
Build directly using cmake for the Cmakelists file found in the project Final_Project
This should automatically link with the libraries in the above project structure
Since we are using openmp, the cmake lists will add the required flags automatically

Details about our implementations can be found in the Implementation_Diary.txt file
If you have any questions or run into any problems, you can contact: sunhaoyang_china@sina.com

## **EntryPointFile:**
main.cpp:
	Input:
1: Camera intrinsic matrix 
2: path to the dataset
3: path to output folder
		using these three to construct a camera object and a RBG_DStream object
		example is in the main.cpp file

output: 
1: a .off merged point cloud in the output folder
2: optimization time and print time in the console
3: EstimatedPoses.txt in the output folder

## **Detailed Description of what main.cpp does**
calling the Init() of RGB_DStream to init the pipeline
calling the ProcessNextFrame() of RGB_DStream object iteratively will read out frames from the dataset and print out the merged point clouds .off file in the end.
calling the printPose() of RGB_DStream will generate the EstimatedPoses.txt file. 
calling end() of RGB_DStream will deconstruct the created object(if it is needed).

## **EstimatedPoses.txt file:**
Each line of the EstimatedPoses.txt file consists of 12 float values. 
They are: (qx qy qz tx ty tz) of the ground truth and the (qx qy qz tx ty tz) of our estimated pose.
The index of the line corresponds to the process order in the stream.

## **Evaluation**
illustration.ipynb is a jupyter notebook script that takes in the file path to the generated EstimatedPoses.txt file and uncomment the specific lines to generate the specific graphs. The instructions are in the notebook file.

Important classes and its functions and variables:
RBG_DStream:
Init() : Sets the setting of our project which includes the number of 
cores for multithreading, Octree settings
ProcessNextFrame(): Reads the frame data, calculates the energy 
function and optimizes the extrinsic of that frame
PrintPointCloud() : Writing the details of a point cloud for mesh lab. 
This is already called in the ProcessNextFrame() function
printPose(): This 

FrameData: Class that holds all details regarding that class which includes the extrinsics, frame id (the unique id given to a frame for processing: needed for the sliding window approach)

VirtualSensor:
	FRAME_INCREMENT: the interval of sampling of the frames to be 
processed.
	m_maxIdx: the maximum index of the frame that will be processed
	m_frameData: stores the data of the frame that is being processed
m_trajectory: a list of ground truth matrixes, whose index does not match the 
index of the frames


## ** additional project version with OpenVDB**
We created a seperate branch that implements OpenVDB. We could not yet properly merge its functionality on all platforms since its installation was complicated. The zip file containing this version is named BundleAdjustment_OpenVDB.zip.

To build the project:
1. Clone and bootstrap vcpkg as instructed here: https://github.com/microsoft/vcpkg#getting-started,
also integrate install visual studio
2. Install dependencies as instructed here:
https://github.com/AcademySoftwareFoundation/openvdb/tree/v9.1.0#installing-dependencies-boost-tbb-blosc-2 
3. Clone OpenVDB, build its project with CMake, and install to your Libs folder
4. Add .\Libs\OpenVDB\bin to your environment paths
5. Create our project (the OpenVDB version) with CMake

This is the repo that does not fit, please download from here
https://drive.google.com/file/d/11_NsgjY0mmrPow_mw_j1-0JIrwrN1opF/view?usp=sharing
Main Contributors: Akbar Suriaganda, Haoyang Sun, Musfira Naqvi, Rohan Fernandez
