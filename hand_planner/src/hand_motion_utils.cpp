#include "hand_motion_utils.h"
#include <ros/ros.h>
#include <cmath>
#include <limits>

static inline RowVectorXd coefRow(const MatrixXd& M, int seg) {
    if (M.rows() == 0 || M.cols() == 0) return RowVectorXd::Zero(6);
    int r = std::max(0, std::min(seg, (int)M.rows()-1));
    RowVectorXd row = M.row(r);
    if (row.size() >= 6) return row.tail(6);
    RowVectorXd out(6); out.setZero(); out.tail(row.size()) = row;
    return out;
}

bool approachWhiteboard(S5_hand& hand, MinimumJerkInterpolation& mj,
                        VectorXd& q, const Vector3d& r_target, const Matrix3d& R_target,
                        double T, const std::function<void(const VectorXd&)>& publish_cb)
{
    MatrixXd t_r(1,4); t_r << 0.0, 3.0, 6.0, 9.0;
    Vector3d r_mid1(0.25, -0.15, -0.33);
    Vector3d r_mid2(0.38,  0.02,  0.03);

    hand.HO_FK_palm(q);
    Vector3d r0 = hand.r_palm;

    MatrixXd Px(1,4), Py(1,4), Pz(1,4);
    Px << r0(0), r_mid1(0), r_mid2(0), r_target(0);
    Py << r0(1), r_mid1(1), r_mid2(1), r_target(1);
    Pz << r0(2), r_mid1(2), r_mid2(2), r_target(2);

    const double INF = std::numeric_limits<double>::infinity();
    MatrixXd Vx(1,4), Vy(1,4), Vz(1,4), Ax(1,4), Ay(1,4), Az(1,4);
    Vx << 0, INF, 0, 0; Vy << 0, INF, 0, 0; Vz << 0, INF, 0, 0;
    Ax << 0, INF, 0, 0; Ay << 0, INF, 0, 0; Az << 0, INF, 0, 0;

    MatrixXd ord(1,3); ord.fill(5);
    MatrixXd conx(3,4), cony(3,4), conz(3,4);
    conx << Px, Vx, Ax; cony << Py, Vy, Ay; conz << Pz, Vz, Az;

    MatrixXd Xc = mj.Coefficient1(t_r, ord, conx, 0.1).transpose();
    MatrixXd Yc = mj.Coefficient1(t_r, ord, cony, 0.1).transpose();
    MatrixXd Zc = mj.Coefficient1(t_r, ord, conz, 0.1).transpose();

    int M = (int)(t_r(0,3)/T);
    ros::Rate rate(1.0/T);

    for (int k=0;k<M;++k){
        double tt = k*T;
        int seg = (tt < t_r(0,1)) ? 0 : ((tt < t_r(0,2)) ? 1 : 2);

        Vector3d V;
        V << mj.GetAccVelPos(coefRow(Xc,seg), tt, 0, 5)(0,1),
             mj.GetAccVelPos(coefRow(Yc,seg), tt, 0, 5)(0,1),
             mj.GetAccVelPos(coefRow(Zc,seg), tt, 0, 5)(0,1);

        hand.update_hand(q, V, r_target, R_target);
        hand.doQP(q);
        q = hand.q_next;

        publish_cb(q);
        ros::spinOnce();
        rate.sleep();
    }
    return true;
}

bool approachViaOneMid(S5_hand& hand, MinimumJerkInterpolation& mj,
                       VectorXd& q, const Vector3d& mid, const Vector3d& goal, const Matrix3d& R_goal,
                       double T1, double T2, double T, const std::function<void(const VectorXd&)>& publish_cb)
{
    hand.HO_FK_palm(q);
    Vector3d r0 = hand.r_palm;

    MatrixXd t(1,3); t << 0.0, T1, T1+T2;
    MatrixXd Px(1,3), Py(1,3), Pz(1,3);
    Px << r0(0), mid(0), goal(0);
    Py << r0(1), mid(1), goal(1);
    Pz << r0(2), mid(2), goal(2);

    const double INF = std::numeric_limits<double>::infinity();
    MatrixXd Vx(1,3), Vy(1,3), Vz(1,3), Ax(1,3), Ay(1,3), Az(1,3);
    Vx << 0, INF, 0; Vy << 0, INF, 0; Vz << 0, INF, 0;
    Ax << 0, INF, 0; Ay << 0, INF, 0; Az << 0, INF, 0;

    MatrixXd ord(1,2); ord.fill(5);
    MatrixXd conx(3,3), cony(3,3), conz(3,3);
    conx << Px, Vx, Ax; cony << Py, Vy, Ay; conz << Pz, Vz, Az;

    MatrixXd Xc = mj.Coefficient1(t, ord, conx, 0.1).transpose();
    MatrixXd Yc = mj.Coefficient1(t, ord, cony, 0.1).transpose();
    MatrixXd Zc = mj.Coefficient1(t, ord, conz, 0.1).transpose();

    int M = (int)((T1+T2)/T);
    ros::Rate rate(1.0/T);

    for (int k=0;k<M;++k){
        double tt = k*T;
        int seg = (tt < T1) ? 0 : 1;

        Vector3d V;
        V << mj.GetAccVelPos(coefRow(Xc,seg), tt, 0, 5)(0,1),
             mj.GetAccVelPos(coefRow(Yc,seg), tt, 0, 5)(0,1),
             mj.GetAccVelPos(coefRow(Zc,seg), tt, 0, 5)(0,1);

        hand.update_hand(q, V, goal, R_goal);
        hand.doQP(q);
        q = hand.q_next;

        publish_cb(q);
        ros::spinOnce();
        rate.sleep();
    }
    return true;
}

bool moveRelative(S5_hand& hand, MinimumJerkInterpolation& mj,
                  VectorXd& q, const Vector3d& dxyz, const Vector3d& goal, const Matrix3d& R_goal,
                  double duration, double T, const std::function<void(const VectorXd&)>& publish_cb)
{
    if (duration < 0.2) duration = 0.2;

    hand.HO_FK_palm(q);
    Vector3d r0 = hand.r_palm;

    MatrixXd t(1,2); t << 0.0, duration;
    MatrixXd Px(1,2), Py(1,2), Pz(1,2);
    Px << r0(0), r0(0)+dxyz(0);
    Py << r0(1), r0(1)+dxyz(1);
    Pz << r0(2), r0(2)+dxyz(2);

    MatrixXd Vx(1,2), Vy(1,2), Vz(1,2), Ax(1,2), Ay(1,2), Az(1,2);
    Vx << 0,0; Vy << 0,0; Vz << 0,0; Ax << 0,0; Ay << 0,0; Az << 0,0;

    MatrixXd ord(1,1); ord.fill(5);
    MatrixXd conx(3,2), cony(3,2), conz(3,2);
    conx << Px, Vx, Ax; cony << Py, Vy, Ay; conz << Pz, Vz, Az;

    MatrixXd Xc = mj.Coefficient1(t, ord, conx, 0.1).transpose();
    MatrixXd Yc = mj.Coefficient1(t, ord, cony, 0.1).transpose();
    MatrixXd Zc = mj.Coefficient1(t, ord, conz, 0.1).transpose();

    int M = (int)(duration/T);
    ros::Rate rate(1.0/T);

    for (int k=0;k<M;++k){
        double tt = k*T;
        Vector3d V;
        V << mj.GetAccVelPos(coefRow(Xc,0), tt, 0, 5)(0,1),
             mj.GetAccVelPos(coefRow(Yc,0), tt, 0, 5)(0,1),
             mj.GetAccVelPos(coefRow(Zc,0), tt, 0, 5)(0,1);

        hand.update_hand(q, V, goal, R_goal);
        hand.doQP(q);
        q = hand.q_next;

        publish_cb(q);
        ros::spinOnce();
        rate.sleep();
    }
    return true;
}

void writeStringCore(S5_hand& hand, VectorXd& q,
                     const std::string& text, const Vector3d& r_target, const Matrix3d& R_target,
                     double T, const std::function<void(const VectorXd&)>& publish_cb)
{
    handWriting hw;
    const double CHAR_T = 5.0, GAP_T = 2.0;
    int N_CHAR = (int)(CHAR_T/T);
    int N_GAP  = (int)(GAP_T/T);

    auto step = [&](double vx,double vy,double vz){
        Vector3d V(vx, -vy, vz);
        hand.update_hand(q, V, r_target, R_target);
        hand.doQP(q);
        q = hand.q_next;
        publish_cb(q);
    };

    char first=0; for(char c:text){ if(!isspace((unsigned char)c)){ first=c; break; } }
    if (first) {
        hw.t = 0.0; hw.T = CHAR_T;
        std::string s(1, std::toupper((unsigned char)first));
        for (int k=0;k<N_CHAR;++k){ hw.move2next(s,s); step(hw.V_x, hw.V_y, hw.V_z); ros::Duration(T).sleep(); }
    }

    auto callLetter = [&](char ch){
        switch (std::tolower((unsigned char)ch)) {
            case 'a': hw.Write_A(CHAR_T); break; case 'b': hw.Write_B(CHAR_T); break; case 'c': hw.Write_C(CHAR_T); break;
            case 'd': hw.Write_D(CHAR_T); break; case 'e': hw.Write_E(CHAR_T); break; case 'f': hw.Write_F(CHAR_T); break;
            case 'g': hw.Write_G(CHAR_T); break; case 'h': hw.Write_H(CHAR_T); break; case 'i': hw.Write_I(CHAR_T); break;
            case 'j': hw.Write_J(CHAR_T); break; case 'k': hw.Write_K(CHAR_T); break; case 'l': hw.Write_L(CHAR_T); break;
            case 'm': hw.Write_M(CHAR_T); break; case 'n': hw.Write_N(CHAR_T); break; case 'o': hw.Write_O(CHAR_T); break;
            case 'p': hw.Write_P(CHAR_T); break; case 'q': hw.Write_Q(CHAR_T); break; case 'r': hw.Write_R(CHAR_T); break;
            case 's': hw.Write_S(CHAR_T); break; case 't': hw.Write_T(CHAR_T); break; case 'u': hw.Write_U(CHAR_T); break;
            case 'v': hw.Write_V(CHAR_T); break; case 'w': hw.Write_W(CHAR_T); break; case 'x': hw.Write_X(CHAR_T); break;
            case 'y': hw.Write_Y(CHAR_T); break; case 'z': hw.Write_Z(CHAR_T); break;
            default: break;
        }
    };

    for (size_t i=0;i<text.size();++i){
        char ch=text[i]; if (isspace((unsigned char)ch)) continue;
        hw.t=0.0; hw.T=CHAR_T;
        for (int k=0;k<N_CHAR;++k){ callLetter(ch); step(hw.V_x, hw.V_y, hw.V_z); ros::Duration(T).sleep(); }

        size_t j=i+1; while (j<text.size() && isspace((unsigned char)text[j])) ++j;
        if (j<text.size()) {
            char nx=text[j];
            hw.t=0.0; hw.T=GAP_T;
            std::string cs(1, std::toupper((unsigned char)ch));
            std::string ns(1, std::toupper((unsigned char)nx));
            for (int k=0;k<N_GAP;++k){ hw.move2next(cs,ns); step(hw.V_x, hw.V_y, hw.V_z); ros::Duration(T).sleep(); }
        }
    }

    hw.t=0.0; hw.T=5.0;
    int N1=(int)(5.0/T);
    for(int k=0;k<N1;++k){ hw.moveback(); step(hw.V_x, hw.V_y, hw.V_z); ros::Duration(T).sleep(); }

    VectorXd q0 = q;
    auto quintic = [](double s){ return s*(s*(10.0 + s*(-15.0 + 6.0*s))); };
    int N2=(int)(5.0/T);
    for(int k=0;k<N2;++k){
        double s = (double)k/N2;
        VectorXd qn(7);
        for(int j=0;j<7;++j) qn(j) = q0(j) + (0.0 - (q(j)-q0(j))) * quintic(s);
        q = qn; publish_cb(q); ros::Duration(T).sleep();
    }
}

void goHome(S5_hand& hand, VectorXd& q,
            const VectorXd& q_home, double T_home, double T,
            const std::function<void(const VectorXd&)>& publish_cb)
{
    VectorXd q_from = q;
    VectorXd q_to   = q_home;
    int STEPS = (int)(T_home/T);
    auto quintic = [](double s){ return s*(s*(10.0 + s*(-15.0 + 6.0*s))); };
    ros::Rate rate(1.0/T);
    for (int k=0;k<=STEPS;++k){
        double s = std::max(0.0, std::min(1.0, (k*T)/T_home));
        VectorXd qn(7);
        for (int j=0;j<7;++j) qn(j) = q_from(j) + (q_to(j) - q_from(j)) * quintic(s);
        q = qn; publish_cb(q); ros::spinOnce(); rate.sleep();
    }
}

bool readLineWithTimeout(std::string &line, double timeout_sec) {
    termios oldt;
    if (tcgetattr(STDIN_FILENO, &oldt) != 0) return false;
    termios newt = oldt; newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

    line.clear();
    ros::WallTime t0 = ros::WallTime::now();
    while (ros::ok()) {
        if ((ros::WallTime::now() - t0).toSec() > timeout_sec) {
            tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
            return false;
        }
        fd_set set; FD_ZERO(&set); FD_SET(STDIN_FILENO, &set);
        timeval tv{0, 100000};
        int rv = select(STDIN_FILENO+1, &set, NULL, NULL, &tv);
        if (rv > 0 && FD_ISSET(STDIN_FILENO, &set)) {
            char ch; ssize_t n = read(STDIN_FILENO, &ch, 1);
            if (n == 1) {
                if (ch == '\r') continue;
                if (ch == '\n') {
                    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
                    return !line.empty();
                }
                line.push_back(ch);
                if ((ch=='x' || ch=='X') && line.size()==1) {
                    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
                    return true;
                }
            }
        }
    }
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return false;
}