#!/usr/bin/env python3
"""Evaluate Social-STGCNN against *executed*, time-aligned Gazebo positions.
PredictedPeople.header.stamp is the observation time t0, not publish time.
Gazebo ModelStates has no header; receipt's ROS simulation time is recorded.
Outputs: raw CSV for all horizons, per-forecast ADE/FDE, summary printed on shutdown.
"""
import os
import csv
import math
import bisect
from collections import deque
import threading
import rospy
from gazebo_msgs.msg import ModelStates
from social_mpc_planner.msg import PredictedPeople, PredictedPeopleUncertainty

class Evaluator:
    def __init__(self):
        rospy.init_node('human_prediction_evaluator')
        self.directory = os.path.expanduser(rospy.get_param('~output_dir','~/navigation_eval_results/human_prediction'))
        os.makedirs(self.directory, exist_ok=True)
        self.dt = float(rospy.get_param('~prediction_dt',0.4))
        self.N = int(rospy.get_param('~prediction_horizon',12))
        self.prefix = rospy.get_param('~person_prefix','person_')
        self.max_alignment_gap = float(rospy.get_param('~max_alignment_gap',0.06))
        self.samples = deque(maxlen=2400)
        self.lock = threading.RLock()
        self.last_evaluate_time = None
        self.sample_spacing = float(rospy.get_param("~sample_spacing", 0.01))
        self.evaluation_spacing = float(rospy.get_param("~evaluation_spacing", 0.10))
        self.received_forecasts = 0
        self.rejected_forecasts = 0
        self.dropped_forecasts = 0
        self.pending = []
        self.uncertainties = {}
        self.uid = 0
        self.total = 0
        self.ade_sum = 0.0
        self.fde_sum = 0.0
        self.rows = open(os.path.join(self.directory,'prediction_errors.csv'),'w',newline='')
        self.writer = csv.writer(self.rows)
        self.writer.writerow(['forecast_id','person_id','t0_sim','k','target_sim','pred_x','pred_y',
                              'gazebo_x','gazebo_y','error_m','sigma_x','sigma_y','rho','mahalanobis_d2'])
        self.summary = open(os.path.join(self.directory,'forecast_metrics.csv'),'w',newline='')
        self.summary_writer = csv.writer(self.summary)
        self.summary_writer.writerow(['forecast_id','person_id','t0_sim','valid_steps','ADE_m','FDE_m'])
        rospy.Subscriber('/gazebo/model_states', ModelStates,self.on_gazebo,queue_size=100)
        rospy.Subscriber('/human_social_stgcnn_prediction',PredictedPeople,self.on_forecast,queue_size=100)
        rospy.Subscriber('/predicted_people_uncertainty',PredictedPeopleUncertainty,self.on_uncertainty,queue_size=100)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo('Human evaluation output: %s',self.directory)

    def on_gazebo(self, msg):
        # ModelStates has no header. /clock is our common simulation time.
        t = rospy.Time.now().to_sec()
        if t <= 0.0:
            return
        with self.lock:
            if self.samples:
                last = self.samples[-1][0]
                if t < last - 0.1:
                    rospy.logwarn('Gazebo simulation time jumped backwards: %.3f -> %.3f; resetting evaluation.', last, t)
                    self.samples.clear()
                    self.pending.clear()
                    self.last_evaluate_time = None
                elif t <= last:
                    # Duplicate/out-of-order ROS timestamp. Never erase forecasts.
                    return
                elif t - last < self.sample_spacing:
                    return
            people = {}
            for i, name in enumerate(msg.name):
                if name.startswith(self.prefix):
                    try:
                        hid = int(name[len(self.prefix):])
                    except ValueError:
                        continue
                    people[hid] = (float(msg.pose[i].position.x),
                                   float(msg.pose[i].position.y))
            if not people:
                return
            self.samples.append((t, people))
            if (self.last_evaluate_time is None or
                    t - self.last_evaluate_time >= self.evaluation_spacing):
                self.last_evaluate_time = t
                self.finish_ready(t)
                rospy.loginfo_throttle(10, 'Evaluator: forecasts=%d pending=%d completed=%d dropped=%d samples=%d',
                                       self.received_forecasts, len(self.pending), self.total,
                                       self.dropped_forecasts, len(self.samples))

    def on_uncertainty(self, msg):
        key = round(msg.header.stamp.to_sec(), 6)
        with self.lock:
            self.uncertainties[key] = {
                p.id: (list(p.sigma_x), list(p.sigma_y), list(p.rho))
                for p in msg.people
            }
            if len(self.uncertainties) > 2000:
                for key_old in sorted(self.uncertainties)[:1000]:
                    del self.uncertainties[key_old]

    def on_forecast(self, msg):
        with self.lock:
            if msg.horizon != self.N or abs(msg.dt-self.dt) > 1e-4:
                self.rejected_forecasts += 1
                rospy.logwarn_throttle(5, 'Evaluator: incompatible forecast dt=%.3f horizon=%d (expected %.3f, %d)',
                                       msg.dt, msg.horizon, self.dt, self.N)
                return
            t0 = msg.header.stamp.to_sec()
            if t0 <= 0.0:
                return
            self.received_forecasts += 1
            for p in msg.people:
                if len(p.trajectory) != self.N+1:
                    self.rejected_forecasts += 1
                    continue
                self.uid += 1
                self.pending.append((self.uid, int(p.id), t0,
                    [(float(v.x), float(v.y)) for v in p.trajectory]))

    def interpolate(self,t,hid):
        if len(self.samples)<2:return None
        # scan backwards; history is ordered by sim time
        for j in range(len(self.samples)-1,0,-1):
            tb,b=self.samples[j]
            ta,a=self.samples[j-1]
            if ta<=t<=tb:
                if t-ta>self.max_alignment_gap or tb-t>self.max_alignment_gap:return None
                if hid not in a or hid not in b:return None
                alpha=(t-ta)/(tb-ta) if tb>ta else 0.0
                return ((1-alpha)*a[hid][0]+alpha*b[hid][0],(1-alpha)*a[hid][1]+alpha*b[hid][1])
            if tb<t:break
        return None

    def finish_ready(self,now):
        due=[]; remaining=[]
        for entry in self.pending:
            if now>=entry[2]+self.N*self.dt+0.03:due.append(entry)
            else:remaining.append(entry)
        self.pending=remaining
        for uid,hid,t0,traj in due:
            e=[]
            unc=self.uncertainties.get(round(t0,6),{}).get(hid)
            for k in range(1,self.N+1):
                target=t0+k*self.dt
                gt=self.interpolate(target,hid)
                if gt is None:continue
                px,py=traj[k]
                dist=math.hypot(gt[0]-px,gt[1]-py)
                sx=sy=rho=d2=float('nan')
                if unc and k<len(unc[0]):
                    sx,sy,rho=unc[0][k],unc[1][k],unc[2][k]
                    denom=max(1e-12,1-rho*rho)
                    if sx>1e-7 and sy>1e-7:
                        u=(gt[0]-px)/sx;v=(gt[1]-py)/sy
                        d2=(u*u-2*rho*u*v+v*v)/denom
                self.writer.writerow([uid,hid,'%.6f'%t0,k,'%.6f'%target,px,py,gt[0],gt[1],dist,sx,sy,rho,d2]);e.append((k,dist))
            if len(e)==self.N:
                ade=sum(x[1] for x in e)/len(e)
                fde=e[-1][1]
                self.summary_writer.writerow([uid,hid,'%.6f'%t0,len(e),ade,fde])
                self.total+=1;self.ade_sum+=ade;self.fde_sum+=fde
            else:
                self.dropped_forecasts += 1
                rospy.logwarn_throttle(5,'Dropped incomplete forecast %s: %d/%d synchronized points',uid,len(e),self.N)
            self.rows.flush();self.summary.flush()

    def shutdown(self):
        if self.total:
            rospy.loginfo('Full forecasts=%d; mean ADE=%.4fm; mean FDE=%.4fm',
                          self.total,self.ade_sum/self.total,self.fde_sum/self.total)
        rospy.loginfo('Evaluator shutdown: received=%d, completed=%d, dropped=%d, rejected=%d',
                      self.received_forecasts, self.total, self.dropped_forecasts, self.rejected_forecasts)
        self.rows.close();self.summary.close()

if __name__=='__main__':
    Evaluator();rospy.spin()
