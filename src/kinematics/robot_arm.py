import sympy
import numpy as np
from numpy import linalg as LA
from typing import List

class RobotArm():
    
    def __init__(self):
        self.num_joints = 0
        self.alpha = None
        self.a = None
        self.q = None
        self.d = None
        self.dh_parameters = {}
        self.transformation_matrixes_list = []
        self.current_pos = []
    
    def set_joints(self, num_joints: int)-> None:
        if num_joints > 0:
            self.num_joints = int(num_joints)
            self.set_dh_parameters()
        else:
            raise("Number of joints needs to be larger than 0.")

    def set_dh_parameters(self)->None:
        self.alpha = sympy.symbols("alpha0:" + str(self.num_joints))
        self.a = sympy.symbols("a0:" + str(self.num_joints))
        self.q = sympy.symbols("q0:" + str(self.num_joints))
        self.d = sympy.symbols("d0:" + str(self.num_joints))
    
    def show_dh_parameters(self)->None:
        print(f"DH Parameters are: {self.dh_parameters}")
    

    def set_dh_parameters_dict(self, dh_parameters_values: List[List])->None:
        for i in range(len(dh_parameters_values)):
            self.dh_parameters[self.alpha[i]] = dh_parameters_values[i][0]
            self.dh_parameters[self.a[i]] = dh_parameters_values[i][1]

            if dh_parameters_values[i][2] == "D":
                self.dh_parameters[self.d[i]] = self.d[i]
            else:
                self.dh_parameters[self.d[i]] =dh_parameters_values[i][2]

            if dh_parameters_values[i][3] == "R":
                self.dh_parameters[self.q[i]] = self.q[i]
            else:
                self.dh_parameters[self.q[i]] = dh_parameters_values[i][3]
        self.set_transformation_matrixes()
    
    def set_transformation_matrixes(self)->None:
        T = sympy.eye(self.num_joints)
        for j in range(self.num_joints):
            T = T * self.transformation_matrix(self.alpha[j], self.a[j], self.d[j], self.q[j]).subs(self.dh_parameters)
            self.transformation_matrixes_list.append(T)
    
    def show_transformation_matrixes(self)->None:
        print(f"Transformation Matrixes are: {self.transformation_matrixes_list}")

    @staticmethod
    def transformation_matrix(alpha, a, d, q)->sympy.Matrix:
	    tf_matrix = sympy.Matrix([[sympy.cos(q),-sympy.sin(q), 0., a],
	                [sympy.sin(q)*sympy.cos(alpha), sympy.cos(q)*sympy.cos(alpha), -sympy.sin(alpha), -sympy.sin(alpha)*d],
	                [sympy.sin(q)*sympy.sin(alpha), sympy.cos(q)*sympy.sin(alpha), sympy.cos(alpha),  sympy.cos(alpha)*d],
	                [   0.,  0.,  0.,  1.]])
	    return tf_matrix

    def forward_kinematics(self, theta_list:list)->np.array:
        theta_dict = {}

        T_0G = self.transformation_matrixes_list[-1]
        
        for i in range(len(theta_list)):
            theta_dict[self.q[i]] = theta_list[i]

        temp = T_0G.evalf(subs = theta_dict, chop=True, maxn=4)

        x = [np.array(temp[0,-1]).astype(np.float64)]       
        y = [np.array(temp[1,-1]).astype(np.float64)]       
        z = [np.array(temp[2,-1]).astype(np.float64)]
        self.current_pos.append((np.array([x,y,z])))
        return self.current_pos       

    def jacobian(self)->None:
        T_0G = self.transformation_matrixes_list[-1]
        self.jacobian_matrix = [diff(T_0G[:3,-1], self.q[i]).reshape(1,3) for i in range(len(self.q))]
        self.jacobian_matrix = sympy.Matrix(self.jacobian_matrix).T

    def inverse_kinematics(self, guess, target)->np.array:
        error = 1.0
        tolerance = 0.5
        
        # Initial guess - Joint angles
        Q = guess
        # X, Y expression
        # X, Y value for target position
        target = np.matrix(target)
        print(target.shape)
        # Jacobian
        self.jacobian()
        T_0G = self.transformation_matrixes_list[-1]

        error_grad = []
        theta_dict = {}

        lr = 0.2
        while error > tolerance:
            for i in range(len(Q)):
                theta_dict[self.q[i]] = Q[i]
            T_q = np.matrix(self.forward_kinematics(Q)[-1])

            delta_T = target - T_q
            Q = Q + lr * (np.matrix(self.jacobian_matrix.evalf(subs = theta_dict, chop = True, maxn = 4)).astype(np.float64).T * delta_T).T
            Q = np.array(Q)[0]
            prev_error = error
            error = LA.norm(delta_T)

            if error > 10 * tolerance:
                lr = 0.3
            elif error < 10 * tolerance:
                lr = 0.02
            error_grad.append((error - prev_error))
        return Q


if __name__ == "__main__":
    arm = RobotArm()
    arm.set_joints(4)
    arm.set_dh_parameters_dict([[0,0,1,'R'],[-np.pi/2.,0,0,'R'],[0,1,0,'R'],[0,1,0,'R']])
    arm.show_dh_parameters()
    tf_matrixes = arm.transformation_matrixes_list
    theta_list = [0, np.pi/2, np.pi, np.pi/3]
    print(arm.forward_kinematics(theta_list))# Seems to work, test with different dh parameters
    #TODO: setup forward and inverse for example like in INN paper
    #TODO: visualize