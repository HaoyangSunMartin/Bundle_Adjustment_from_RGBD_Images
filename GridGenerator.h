#pragma once

#include "VirtualSensor.h"
#include <limits>

using namespace std;
using namespace Eigen;

class Grid {
protected:
	size_t width, height, depth;
	Vector3f minCoord, maxCoord, minOpt, maxOpt, cellSize;
	bool printOpt = true;

	void Setup(int width, int height, int depth, Vector3f minCoord, Vector3f maxCoord) {
		this->width = width;
		this->height = height;
		this->depth = depth;

		this->minCoord = minCoord;
		this->maxCoord = maxCoord;
		minOpt.x() = minOpt.y() = minOpt.z() = numeric_limits<float>().max();
		maxOpt.x() = maxOpt.y() = maxOpt.z() = numeric_limits<float>().lowest();

		cellSize.x() = (maxCoord.x() - minCoord.x()) / width;
		cellSize.y() = (maxCoord.y() - minCoord.y()) / height;
		cellSize.z() = (maxCoord.z() - minCoord.z()) / depth;
	}

	bool valid_point(Vertex* vertices, unsigned int idx) {
		return vertices[idx].position != Vector4f(MINF, MINF, MINF, MINF);

	}

	size_t ToCoord(size_t x, size_t y, size_t z) {
		return x + width * y + depth * width * z;
	}

	size_t ToCoord(Vector3f point) {
		size_t x = (size_t)min(max(0.0f, round((point.x() - minCoord.x()) / cellSize.x())), width - 1.0f);
		size_t y = (size_t)min(max(0.0f, round((point.y() - minCoord.y()) / cellSize.y())), height - 1.0f);
		size_t z = (size_t)min(max(0.0f, round((point.z() - minCoord.z()) / cellSize.z())), depth - 1.0f);

		return ToCoord(x, y, z);
	}

	Vector3f ToPos(size_t coord) {
		size_t z = (size_t)(coord / (depth * width));
		size_t y = (size_t)((coord - z * (depth * width)) / width);
		size_t x = (size_t)(coord - z * (depth * width) - y * width);
		return ToPos(x, y, z);
	}

	Vector3f ToPos(size_t x, size_t y, size_t z) {
		Vector3f pos;
		pos.x() = minCoord.x() + x * cellSize.x();
		pos.y() = minCoord.y() + y * cellSize.y();
		pos.z() = minCoord.z() + z * cellSize.z();

		return pos;
	}

	void UpdateLimits(Vector3f point) {
		minOpt.x() = min(minOpt.x(), point.x());
		minOpt.y() = min(minOpt.y(), point.y());
		minOpt.z() = min(minOpt.z(), point.z());
		maxOpt.x() = max(maxOpt.x(), point.x());
		maxOpt.y() = max(maxOpt.y(), point.y());
		maxOpt.z() = max(maxOpt.z(), point.z());
	}

	size_t GetSize() {
		return width * height * depth;
	}

	void PrintOpt() {
		if (printOpt)
			cout << "=== Grid updated:"
			<< "\noptimal min: " << minOpt.x() << " " << minOpt.y() << " " << minOpt.z()
			<< "\noptimal max: " << maxOpt.x() << " " << maxOpt.y() << " " << maxOpt.z() << endl;
	}
};

class OccupancyGrid : Grid {
public:
	OccupancyGrid(int l, Vector3f minCoord, Vector3f maxCoord) {
		Setup(l, l, l, minCoord, maxCoord);
		SetupColors();
	}
	OccupancyGrid(int width, int height, int depth, Vector3f minCoord, Vector3f maxCoord) {
		Setup(width, height, depth, minCoord, maxCoord);
		SetupColors();
	}

	~OccupancyGrid() {
		if (grid)
			delete grid;
		if (colors)
			delete colors;
	}

	/// <summary>
	/// Fill the grid with new points.
	/// </summary>
	/// <param name="vertices">added points</param>
	/// <param name="extrinsics">pose of these points</param>
	/// <param name="size">amount of points</param>
	void InsertPoints(Vertex* vertices, Matrix4f extrinsics, size_t size) {
		//#pragma omp parallel for
		for (int i = 0; i < size; i++)
		{
			// check first if it's a valid point
			if (valid_point(vertices, i)) {
				// get pos in world space
				Vector3f pos = (extrinsics * vertices[i].position).head<3>();
				UpdateLimits(pos);
				size_t coord = ToCoord(pos);
				// add to grid
				size_t amount = grid[coord];
				grid[coord]++;
				Vector4f color;
				color << vertices[i].color(0) * 1.0f, vertices[i].color(1) * 1.0f, vertices[i].color(2) * 1.0f, vertices[i].color(3) * 1.0f;
				//float blend = 1.0f / (amount + 1.0f);
				//colors[coord] = (1 - blend) * colors[coord] + blend * color;
				colors[coord] = color;
			}
		}

		PrintOpt();
	}

	/// <summary>
	/// Remove the points from the grid.
	/// If the point was the only one in the cell, the cell will be empty again.
	/// </summary>
	/// <param name="vertices">removing points</param>
	/// <param name="extrinsics">pose of these points</param>
	/// <param name="size">amount of points</param>
	void RemovePoints(Vertex* vertices, Matrix4f extrinsics, size_t size) {
		for (int i = 0; i < size; i++)
		{
			// check first if it's a valid point
			if (valid_point(vertices, i)) {
				// get pos in world space
				Vector3f pos = (extrinsics * vertices[i].position).head<3>();
				size_t coord = ToCoord(pos);
				// remove from grid
				size_t amount = grid[coord];
				if (amount <= 1) {
					grid[coord] = 0;
					// reset color if cell is empty;
					colors[coord] = Vector4f::Ones();
				}
				else {
					grid[coord]--;
				}
			}
		}
	}

	/// <summary>
	/// Create new file with all the points in the grid.
	/// </summary>
	/// <param name="filename">output file</param>
	/// <param name="size">size of vertex array</param>
	/// <returns>false if file could not be written</returns>
	bool WriteMesh(const string& filename, const size_t imageSize) {
		// check first if possible to save
		std::ofstream outFile;
		outFile.open(filename, std::ios_base::trunc);
		if (!outFile.is_open()) {
			std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
			return false;
		}

		outFile << "COFF" << std::endl;
		outFile << "# numVertices numFaces numEdges" << std::endl;

		size_t nVertices = 0;
		size_t size = GetSize();
		for (size_t i = 0; i < size; i++)
		{
			if (grid[i] > 0) nVertices++;
		}
		outFile << nVertices << " 0 0" << std::endl;

		// save vertices
		// iterate through the grid
//#pragma omp parallel for
		for (int i = 0; i < size; i++)
		{
			// check if cell has colors
			if (grid[i] == 0) continue;

			// calculate position
			Vector3f pos = ToPos(i);

			// get color
			Vector4f color = colors[i];

			// print line
			outFile << pos.x() << " " << pos.y() << " " << pos.z() << " "
				<< color(0) << " " << color(1) << " " << color(2) << " " << color(3) << endl;
		}


		outFile.close();

		return true;
	}

private:
	size_t* grid;
	Vector4f* colors;

	void SetupColors() {
		size_t size = GetSize();
		grid = new size_t[size];
		colors = new Vector4f[size];
		//#pragma omp parallel for
		for (size_t i = 0; i < size; i++)
		{
			grid[i] = 0;
			colors[i] = Vector4f::Ones();
		}
	}
};

// Implementation of Octree in c++
// copied from https://www.geeksforgeeks.org/octree-insertion-and-searching/ and modified for our context

#define TopLeftFront 0
#define TopRightFront 1
#define BottomRightFront 2
#define BottomLeftFront 3
#define TopLeftBack 4
#define TopRightBack 5
#define BottomRightBack 6
#define BottomLeftBack 7

// Structure of a point
class OctreePoint {
public:
	int x;
	int y;
	int z;

	size_t amount;
	Vector4f color;

	OctreePoint()
		: x(-1), y(-1), z(-1), amount(0), color(Vector4f::Zero())
	{
	}

	OctreePoint(int a, int b, int c)
		: x(a), y(b), z(c), amount(0), color(Vector4f::Zero())
	{
	}

	OctreePoint(int a, int b, int c, Vector4f color)
		: x(a), y(b), z(c), amount(0)
	{
		AddColor(color);
	}

	void AddColor(Vector4f color) {
		//amount++;
		//float blend = 1.0f / amount;
		//this->color = (1 - blend) * this->color + blend * color;
		this->color = color;
	}
};

// Octree class
class Octree : Grid {
private:
	// if point == NULL, node is internal node.
	// if point == (-1, -1, -1), node is empty.
	OctreePoint* point;

	// Represent the boundary of the cube
	vector<Octree*> children;

	void SetupBoundaries(int x1, int y1, int z1, int x2, int y2, int z2)
	{
		// This use to construct Octree
		// with boundaries defined
		if (x2 < x1
			|| y2 < y1
			|| z2 < z1) {
			cout << "boundary points are not valid" << endl;
			return;
		}

		point = nullptr;
		topLeftFront
			= new OctreePoint(x1, y1, z1);
		bottomRightBack
			= new OctreePoint(x2, y2, z2);

		// Assigning null to the children
		if (x1 != x2) {
			children.assign(8, nullptr);
			for (int i = 0; i < 8; ++i)
				children[i] = new Octree();
		}
	}

public:
	OctreePoint* topLeftFront, * bottomRightBack;

	// Constructor
	Octree()
	{
		// To declare empty node
		point = new OctreePoint();

	}

	// Constructor with three arguments
	Octree(int x, int y, int z, Vector4f color)
	{
		point = new OctreePoint(x, y, z, color);
	}

	// Constructor with six arguments
	Octree(size_t x1, size_t y1, size_t z1, size_t x2, size_t y2, size_t z2)
	{
		size_t w = x2 - x1;
		size_t h = y2 - y1;
		size_t d = z2 - z1;
		Vector3f minCoord = ToPos(x1, y1, z1);
		Vector3f maxCoord = ToPos(x2, y2, z2);

		Setup(w, h, d, minCoord, maxCoord);
		SetupBoundaries(x1, y1, z1, x2, y2, z2);
	}

	Octree(int l, Vector3f minCoord, Vector3f maxCoord) {
		Setup(l, l, l, minCoord, maxCoord);

		int maxValue = l - 1;
		SetupBoundaries(0, 0, 0, maxValue, maxValue, maxValue);
	}

	/// <summary>
	/// Fill the grid with new points.
	/// </summary>
	/// <param name="vertices">added points</param>
	/// <param name="extrinsics">pose of these points</param>
	/// <param name="size">amoint of points</param>
	void InsertPoints(Vertex* vertices, Matrix4f extrinsics, size_t size) {
		for (size_t i = 0; i < size; i++)
		{
			// check first if it's a valid point
			if (valid_point(vertices, i)) {
				// get pos in world space
				Vector3f pos = (extrinsics * vertices[i].position).head<3>();
				UpdateLimits(pos);

				// get coordinates
				size_t coord = ToCoord(pos);
				size_t z = (size_t)(coord / (depth * width));
				size_t y = (size_t)((coord - z * (depth * width)) / width);
				size_t x = (size_t)(coord - z * (depth * width) - y * width);
				// add to grid
				Vector4f color;
				color << vertices[i].color(0) * 1.0f, vertices[i].color(1) * 1.0f, vertices[i].color(2) * 1.0f, vertices[i].color(3) * 1.0f;
				insert(x, y, z, color);
			}
		}

		PrintOpt();
	}

	// Function to insert a point in the octree
	void insert(int x, int y, int z, Vector4f color)
	{

		// If the point already exists in the octree
		OctreePoint* point = find(x, y, z);
		if (point) {
			point->AddColor(color);
			cout << "Point already exist in the tree" << endl;
			return;
		}

		// If the point is out of bounds
		if (x < topLeftFront->x
			|| x > bottomRightBack->x
			|| y < topLeftFront->y
			|| y > bottomRightBack->y
			|| z < topLeftFront->z
			|| z > bottomRightBack->z) {
			cout << "Point is out of bound" << endl;
			return;
		}

		// Binary search to insert the point
		int midx = (topLeftFront->x
			+ bottomRightBack->x)
			/ 2;
		int midy = (topLeftFront->y
			+ bottomRightBack->y)
			/ 2;
		int midz = (topLeftFront->z
			+ bottomRightBack->z)
			/ 2;

		int pos = -1;

		// Checking the octant of
		// the point
		if (x <= midx) {
			if (y <= midy) {
				if (z <= midz)
					pos = TopLeftFront;
				else
					pos = TopLeftBack;
			}
			else {
				if (z <= midz)
					pos = BottomLeftFront;
				else
					pos = BottomLeftBack;
			}
		}
		else {
			if (y <= midy) {
				if (z <= midz)
					pos = TopRightFront;
				else
					pos = TopRightBack;
			}
			else {
				if (z <= midz)
					pos = BottomRightFront;
				else
					pos = BottomRightBack;
			}
		}

		// If an internal node is encountered
		if (children[pos]->point == nullptr) {
			children[pos]->insert(x, y, z, color);
			return;
		}

		// If an empty node is encountered
		else if (children[pos]->point->x == -1) {
			delete children[pos];
			children[pos] = new Octree(x, y, z, color);
			return;
		}

		// leaf reached
		else if (topLeftFront->x == midx) {
			children[pos]->point->AddColor(color);
			return;
		}

		else {
			int x_ = children[pos]->point->x,
				y_ = children[pos]->point->y,
				z_ = children[pos]->point->z;
			Vector4f storedColor = children[pos]->point->color;
			delete children[pos];
			children[pos] = nullptr;
			if (pos == TopLeftFront) {
				children[pos] = new Octree(topLeftFront->x,
					topLeftFront->y,
					topLeftFront->z,
					midx,
					midy,
					midz);
			}

			else if (pos == TopRightFront) {
				children[pos] = new Octree(midx + 1,
					topLeftFront->y,
					topLeftFront->z,
					bottomRightBack->x,
					midy,
					midz);
			}
			else if (pos == BottomRightFront) {
				children[pos] = new Octree(midx + 1,
					midy + 1,
					topLeftFront->z,
					bottomRightBack->x,
					bottomRightBack->y,
					midz);
			}
			else if (pos == BottomLeftFront) {
				children[pos] = new Octree(topLeftFront->x,
					midy + 1,
					topLeftFront->z,
					midx,
					bottomRightBack->y,
					midz);
			}
			else if (pos == TopLeftBack) {
				children[pos] = new Octree(topLeftFront->x,
					topLeftFront->y,
					midz + 1,
					midx,
					midy,
					bottomRightBack->z);
			}
			else if (pos == TopRightBack) {
				children[pos] = new Octree(midx + 1,
					topLeftFront->y,
					midz + 1,
					bottomRightBack->x,
					midy,
					bottomRightBack->z);
			}
			else if (pos == BottomRightBack) {
				children[pos] = new Octree(midx + 1,
					midy + 1,
					midz + 1,
					bottomRightBack->x,
					bottomRightBack->y,
					bottomRightBack->z);
			}
			else if (pos == BottomLeftBack) {
				children[pos] = new Octree(topLeftFront->x,
					midy + 1,
					midz + 1,
					midx,
					bottomRightBack->y,
					bottomRightBack->z);
			}
			children[pos]->insert(x_, y_, z_, storedColor);
			children[pos]->insert(x, y, z, color);
		}
	}

	// Function that returns true if the point
	// (x, y, z) exists in the octree
	OctreePoint* find(int x, int y, int z)
	{
		// If point is out of bound
		if (x < topLeftFront->x
			|| x > bottomRightBack->x
			|| y < topLeftFront->y
			|| y > bottomRightBack->y
			|| z < topLeftFront->z
			|| z > bottomRightBack->z)
			return 0;

		// Otherwise perform binary search
		// for each ordinate
		int midx = (topLeftFront->x
			+ bottomRightBack->x)
			/ 2;
		int midy = (topLeftFront->y
			+ bottomRightBack->y)
			/ 2;
		int midz = (topLeftFront->z
			+ bottomRightBack->z)
			/ 2;

		int pos = -1;

		// Deciding the position
		// where to move
		if (x <= midx) {
			if (y <= midy) {
				if (z <= midz)
					pos = TopLeftFront;
				else
					pos = TopLeftBack;
			}
			else {
				if (z <= midz)
					pos = BottomLeftFront;
				else
					pos = BottomLeftBack;
			}
		}
		else {
			if (y <= midy) {
				if (z <= midz)
					pos = TopRightFront;
				else
					pos = TopRightBack;
			}
			else {
				if (z <= midz)
					pos = BottomRightFront;
				else
					pos = BottomRightBack;
			}
		}

		// If an internal node is encountered
		if (children[pos]->point == nullptr) {
			return children[pos]->find(x, y, z);
		}

		// If an empty node is encountered
		else if (children[pos]->point->x == -1) {
			return 0;
		}
		else {

			// If node is found with
			// the given value
			if (x == children[pos]->point->x
				&& y == children[pos]->point->y
				&& z == children[pos]->point->z)
				return point;
		}
		return 0;
	}


	/// <summary>
	/// Create new file with all the points in the grid.
	/// </summary>
	/// <param name="filename">output file</param>
	/// <param name="size">size of vertex array</param>
	/// <returns>false if file could not be written</returns>
	bool WriteMesh(const string& filename) {
		// check first if possible to save
		std::ofstream outFile;
		outFile.open(filename, std::ios_base::trunc);
		if (!outFile.is_open()) {
			std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
			return false;
		}

		outFile << "COFF" << std::endl;
		outFile << "# numVertices numFaces numEdges" << std::endl;

		vector<OctreePoint> points;
		GetPoints(points);

		size_t nVertices = points.size();
		outFile << nVertices << " 0 0" << std::endl;

		// save vertices
		// iterate through the grid
		for (int i = 0; i < points.size(); i++)
		{
			OctreePoint op = points[i];
			// calculate position
			Vector3f pos = ToPos(op.x, op.y, op.z);

			// get color
			Vector4f color = op.color;

			// print line
			outFile << pos.x() << " " << pos.y() << " " << pos.z() << " "
				<< color(0) << " " << color(1) << " " << color(2) << " " << color(3) << endl;
		}


		outFile.close();

		return true;
	}

	void GetPoints(vector<OctreePoint>& points) {
		// leaf reached
		if (children.size() == 0 && point->x >= 0) {
			points.push_back(*point);
			return;
		}

		for (size_t i = 0; i < children.size(); i++)
		{
			children[i]->GetPoints(points);
		}
	}
};
