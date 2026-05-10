#ifndef HANDWRITING_H
#define HANDWRITING_H
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <map>
using namespace std;

class handWriting
{
public:
    double a_x;
    double a_y;
    double a_z;
    double a;
    double b;
    double c;
    double d;
    double P_start;
    double P_end;
    double dt;
    double t;
    double T;
    double L;
    double V;
    double P;
    double V_x;
    double V_y;
    double V_z;
    map<string,double>Z_start;
    map<string,double>Y_start;
    map<string,double>Z_end;
    map<string,double>Y_end;
    handWriting();
    double saturate(double a, double min, double max);
    double move2pose(double max, double t_local, double T_start, double T_end);
    double move_to_diff(double max, double t_local, double T_start, double T_end);
    double Velocity(double _L, double _t, double _T);
    double Position(double _L, double _t, double _T);
    void move2next(string current, string next);
    double move2zero(double theta,double t,double T_home);
    void defineLetter(string letter, double z_start, double y_start, double z_end, double y_end);

    void Write_A(double T);
    void Write_B(double T);
    void Write_C(double T);
    void Write_D(double T);
    void Write_E(double T);
    void Write_F(double T);
    void Write_G(double T);
    void Write_H(double T);
    void Write_I(double T);
    void Write_J(double T);
    void Write_K(double T);
    void Write_L(double T);
    void Write_M(double T);
    void Write_N(double T);
    void Write_O(double T);
    void Write_P(double T);
    void Write_Q(double T);
    void Write_R(double T);
    void Write_S(double T);
    void Write_T(double T);
    void Write_U(double T);
    void Write_V(double T);
    void Write_W(double T);
    void Write_X(double T);
    void Write_Y(double T);
    void Write_Z(double T);
    
    void Write_x_axis(double a_x ,double T);
    void Write_y_axis(double a_y ,double T);
    void Write_z_axis(double a_y ,double T);



    void moveback();
};

#endif // HANDWRITING_H
