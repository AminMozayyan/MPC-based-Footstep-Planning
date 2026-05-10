#pragma once
#ifndef HAND_MOTION_UTILS_H
#define HAND_MOTION_UTILS_H

#include "S5_hand.h"
#include "MinimumJerkInterpolation.h"
#include "eigen3/Eigen/Dense"
#include <functional>
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
#include <fcntl.h>
#include <string>
#include "handwriting.h"

using namespace Eigen;

bool approachWhiteboard(S5_hand& hand, MinimumJerkInterpolation& mj,
                        VectorXd& q, const Vector3d& r_target, const Matrix3d& R_target,
                        double T, const std::function<void(const VectorXd&)>& publish_cb);

bool approachViaOneMid(S5_hand& hand, MinimumJerkInterpolation& mj,
                       VectorXd& q, const Vector3d& mid, const Vector3d& goal, const Matrix3d& R_goal,
                       double T1, double T2, double T, const std::function<void(const VectorXd&)>& publish_cb);

bool moveRelative(S5_hand& hand, MinimumJerkInterpolation& mj,
                  VectorXd& q, const Vector3d& dxyz, const Vector3d& goal, const Matrix3d& R_goal,
                  double duration, double T, const std::function<void(const VectorXd&)>& publish_cb);

void writeStringCore(S5_hand& hand, VectorXd& q,
                     const std::string& text, const Vector3d& r_target, const Matrix3d& R_target,
                     double T, const std::function<void(const VectorXd&)>& publish_cb);

void goHome(S5_hand& hand, VectorXd& q,
            const VectorXd& q_home, double T_home, double T,
            const std::function<void(const VectorXd&)>& publish_cb);

bool readLineWithTimeout(std::string &line, double timeout_sec);

#endif
