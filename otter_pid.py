
from sre_constants import CATEGORY_LINEBREAK
import sys
import time
import numpy as np
import matplotlib.pylab as plt
import math
from pid import PID
sys.path.insert(0,'~/Desktop/Calismalar/git/otter/otter_simulator/OtterSimulator')
from OtterSimulator.otter import Otter
from OtterSimulator import gnc

class Controller():
    def __init__(self):
        self.saturation_limit = 104.72
        self.Kp=0.0
        self.Kd=0.0
        self.Ki=0.0
        self.Kf=0.0
        self.derivative_feedback_coeff=0.0
        
        self.surge_pid=PID(self.Kp,self.Ki,self.Kd,derivative_feedback=True,derivative_filter_coeff=self.derivative_feedback_coeff,saturation_limit=self.saturation_limit,Kf=self.Kf)
        self.surge_pid.set_setpoint(0.0)
        self.usv_velocity_measured = (0.0,0.0)
        self.vel = 0.0
        self.dt = 0.0
        self.err_cum = 0.0
        self.lin_vel_set = 0.0
        self.ang_vel_set = 0.0
        self.motor_signal=[]

    
    def surge_controller(self,ref_velocity,nu,dt):
        k_pos=0.02216/2
        u_speed=nu[0]
        v_speed=nu[1]
        U_speed=math.sqrt(u_speed**2+v_speed**2)
        self.Kp=20
        self.Kd=25
        self.Ki=1250
        self.Kf=(3.684*ref_velocity**3-23.44*ref_velocity**2+67.35*ref_velocity+12.3)/ref_velocity
        print('*******************************')
        print('Kf:',self.Kf)
        self.surge_pid=PID(self.Kp,self.Ki,self.Kd,derivative_feedback=True,derivative_filter_coeff=self.derivative_feedback_coeff,saturation_limit=self.saturation_limit,Kf=self.Kf)  
        print('U_speed:',U_speed)
        self.surge_pid.set_setpoint(ref_velocity) 
        surge_signals=self.surge_pid.execute(U_speed,dt)
        surge_signals=2*k_pos*surge_signals*abs(surge_signals)
        print('surge_signal:',surge_signals)
        
        return surge_signals

    
    def heading_controller(self,ref_psi,nu,eta,dt):
        k_pos=0.02216/2
        r=nu[5]
        heading=eta[5]
        self.Kp=250
        self.Kd=0
        self.Ki=1500
        self.Kf=0
        self.yaw_pid=PID(self.Kp,self.Ki,self.Kd,derivative_feedback=True,derivative_filter_coeff=self.derivative_feedback_coeff,saturation_limit=self.saturation_limit,Kf=self.Kf)
        self.yaw_pid.set_setpoint(ref_heading)
        yaw_signals=self.yaw_pid.execute(heading,dt)
        yaw_signals=k_pos*yaw_signals*abs(yaw_signals)
        print('yaw_signal:',yaw_signals)
        # yaw_n1 = controller_signal
        # yaw_n2 = controller_signal
        # yaw_signals=[yaw_n1,yaw_n2]
        return yaw_signals

    
    def control_allocation(self,surge_signals,yaw_signals):
        l1 = -0.395
        l2 = 0.395
        k_pos = 0.02216/2
        B=k_pos*np.array([[1,1],[-l1,-l2]])
        B_inv=np.linalg.inv(B)
        signal_vector=np.array([[surge_signals],[yaw_signals]]) 
        u_alloc = np.matmul(B_inv,signal_vector) # u_alloc = inv(B) * signal_vector
        print('u_alloc:',u_alloc)
        n1 = np.sign(u_alloc[0]) * math.sqrt(abs(u_alloc[0]))  # u_alloc = abs(n) * n --> n = sign(u_alloc) * sqrt(u_alloc)
        n2 = np.sign(u_alloc[1]) * math.sqrt(abs(u_alloc[1]))
        
        return n1,n2





if __name__=='__main__':


    start_time=time.time()
    last_time=start_time
    dt=0.1
    vehicle=Otter()
    U_controller=Controller()
    vehicle.nu = np.array([0, 0, 0, 0, 0, 0], float)
    vehicle.u_actual = np.array([0, 0], float)
    vehicle.u_control = np.array([0, 0], float)           
    current_eta = np.array([0, 0, 0, 0, 0, 0], float)     # Vehicle linear and angluar initial position
    ref_velocity=2.5
    ref_heading=0.2
    t=[]
    U_speed_list=[]
    endpoint=10
    m=[]
    X_list=[]
    Y_list=[]
    heading_list=[]
    for i in np.linspace(0,endpoint,10*endpoint):
        # current_time=time.time()
        # dt=current_time-last_time
        m.append(i)
        surge_signals = U_controller.surge_controller(ref_velocity,vehicle.nu,dt)
        yaw_signals   = U_controller.heading_controller(ref_heading,vehicle.nu,current_eta,dt) 
        [n1,n2] = U_controller.control_allocation(surge_signals,yaw_signals)
        n1=float(n1)
        n2=float(n2)
        vehicle.u_control=np.array([n1,n2])
        [nu, u_actual] = vehicle.dynamics(current_eta,vehicle.nu,vehicle.u_actual,vehicle.u_control,dt)

        print('u_actual:',u_actual)
        print('-------------------')
        print('r:',nu[5])

        vehicle.nu=nu
        vehicle.u_actual=u_actual 
        current_eta=gnc.attitudeEuler(current_eta,vehicle.nu,dt)  
        print('heading:',current_eta[5])   
        U_speed_list.append(math.sqrt(vehicle.nu[0]**2+vehicle.nu[1]**2))
        X_list.append(current_eta[0])
        Y_list.append(current_eta[1])
        heading_list.append(current_eta[5])

        
    plt.figure(1)
    plt.axhline(y=ref_velocity, color='r', linestyle='-')
    plt.plot(np.linspace(0,endpoint,10*endpoint),U_speed_list)
    

    plt.figure(2)
    plt.axhline(ref_heading,color='g',linestyle='-')
    plt.plot(np.linspace(0,endpoint,10*endpoint),heading_list)
    plt.show()

