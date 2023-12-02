#pragma once

#include "LossFunction.h"
#include <omp.h>


using namespace std;
using namespace Eigen;

class Optimizer {
public:

	
	Optimizer() {}

	//for saving the ID of the residual block
	std::vector<ceres::ResidualBlockId> residualBlockID;
	
	void estimatePose(const vector<Vector3f>& source, const vector<Vector3f>& target, double* pose) {
		ceres::Problem problem;
		prepareConstraints(source, target, pose, problem);

		ceres::Solver::Options options;
		configureSolver(options);

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		//std::cout << summary.BriefReport() << std::endl;

		finalCost = summary.final_cost;
		terminationType = summary.termination_type;
	}

	double finalCost;
	ceres::TerminationType terminationType;

private:
	void prepareConstraints(const vector<Vector3f>& sourcePoints, const vector<Vector3f>& targetPoints, double* pose, ceres::Problem& problem) const {
		const unsigned nPoints = sourcePoints.size();
		
		for (unsigned i = 0; i < nPoints; ++i) {
			const auto& sourcePoint = sourcePoints[i];
			const auto& targetPoint = targetPoints[i];
			if (!sourcePoint.allFinite() || !targetPoint.allFinite())
				continue;

			ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<PointToPointConstraint, 3, 6>(
				new PointToPointConstraint(sourcePoint, targetPoint)
				);
			auto ID = problem.AddResidualBlock(
				costFunction,
				nullptr, pose
			);
			//residualBlockID.emplace_back(ID);
		}
	}

	void configureSolver(ceres::Solver::Options& options) {
		// Ceres options.
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
		options.use_nonmonotonic_steps = false;
		options.linear_solver_type = ceres::DENSE_QR;
		//options.minimizer_progress_to_stdout = 1;
		options.max_num_iterations = 10;
		options.num_threads = omp_get_max_threads();
	}
};
