#pragma once

#include "Eigen.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

/**
 * Helper methods for writing Ceres cost functions.
 */
template <typename T>
static inline void fillVector(const Vector3f& input, T* output) {
	output[0] = T(input[0]);
	output[1] = T(input[1]);
	output[2] = T(input[2]);
}

/// <summary>
/// Class to calculate the error between a target point and a transformed source point using a given pose.
/// </summary>
class PointToPointConstraint {
public:
	PointToPointConstraint(const Vector3f& sourcePoint, const Vector3f& targetPoint) :
		m_sourcePoint{ sourcePoint },
		m_targetPoint{ targetPoint }
	{}

	/// <summary>
	/// Class to calculate the error between a target point and a transformed source point using a given pose.
	/// </summary>
	/// <param name="pose">the current translation and rotation that is to be checked</param>
	/// <param name="residuals">the error between the points</param>
	/// <returns></returns>
	template <typename T>
	bool operator()(const T* const pose, T* residuals) const {

		// extract the translation and rotation values from pose
		const T* rotation = const_cast<T* const>(pose);
		const T* translation = const_cast<T* const>(pose) + 3;
		
		T temp[3];
		T sourcePoint[3];
		fillVector(m_sourcePoint, sourcePoint);
		ceres::AngleAxisRotatePoint(rotation, sourcePoint, temp);

		T outputPoint[3];

		outputPoint[0] = temp[0] + translation[0];
		outputPoint[1] = temp[1] + translation[1];
		outputPoint[2] = temp[2] + translation[2];

		residuals[0] = outputPoint[0] - T(m_targetPoint[0]);
		residuals[1] = outputPoint[1] - T(m_targetPoint[1]);
		residuals[2] = outputPoint[2] - T(m_targetPoint[2]);

		return true;
	}

private:
	const Vector3f m_sourcePoint;
	const Vector3f m_targetPoint;
};