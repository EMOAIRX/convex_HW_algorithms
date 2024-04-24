import numpy as np
class NormalFunction:
    def __init__(self):
        self.n = 3
        

    def __call__(self, x):
        return (3 - x[0])**2 + 7*(x[1] - x[0]**2)**2 + 9*(x[2] - x[0] - x[1]**2)**2

    def gradient(self, x):
        v0 = -28 * x[0] * (-x[0]**2 + x[1]) + 20 * x[0] + 18 * x[1]**2 - 18 * x[2] - 6
        v1 = -14 * x[0]**2 - 36 * x[1] * (-x[0] - x[1]**2 + x[2]) + 14 * x[1]
        v2 = -18 * x[0] - 18 * x[1]**2 + 18 * x[2]
        return np.array([v0, v1, v2])

    def test():
        pass

class Simulator:
    def __init__(self ,ita = 1e-8):
        self.n = 3
        self.function = NormalFunction()
        self.ita = ita
        self.btls_alpha = 0.3 #backtracing_line_search
        self.btls_beta = 0.9


    def backtracking_line_search(self,x,direction):
        t = 1
        differential = self.f_grad(x)
        while self.f(x + t * direction) > self.f(x) + self.btls_alpha * t * np.dot(differential, direction):
            t = self.btls_beta * t
        return t 
    
    def exact_line_search(self,x,direction):
        right = 1
        right = self.backtracking_line_search(x,direction)
        left = 0
        while right - left > 1e-9:
            mid1 = left + (right - left) / 3
            mid2 = right - (right - left) / 3
            if self.f(x + mid1 * direction) < self.f(x + mid2 * direction):
                right = mid2
            else:
                left = mid1
        return (left + right) / 2

            

    def f(self,x):
        return self.function(x)

    def f_grad(self,x):
        return self.function.gradient(x)

    def DFP_algorihtm(self, start_point):
        aH = np.eye(3)
        tracing_list = []
        x = start_point
        grad = self.f_grad(x)
        step = 0
        while np.linalg.norm(grad) >= self.ita:
            direction = -np.dot(aH, grad)
            alpha = self.exact_line_search(x,direction)
            x = x + alpha * direction
            print(x,direction)
            tracing_list.append(x)
            grad_old = grad
            grad = self.f_grad(x)
            DeltaX = alpha * direction
            DeltaG = grad - grad_old
            #DFP algorithm
            aH = aH + np.outer(DeltaX, DeltaX) / np.dot(DeltaX, DeltaG) - np.outer(np.dot(aH, DeltaG),np.dot(aH, DeltaG)) / np.dot(np.dot(DeltaG, aH), DeltaG)        
        return aH,tracing_list
    def BFGS_algorithm(self,start_point):
        aH = np.eye(3)
        tracing_list = []
        x = start_point
        grad = self.f_grad(x)
        step = 0
        while np.linalg.norm(grad) >= self.ita:
            direction = -np.dot(aH, grad)
            alpha = self.exact_line_search(x,direction)
            x = x + alpha * direction
            print(x,direction)
            tracing_list.append(x)
            grad_old = grad
            grad = self.f_grad(x)
            DeltaX = alpha * direction
            DeltaG = grad - grad_old
            rho = 1 / np.dot(DeltaG, DeltaX)
            V = (np.eye(3) - rho * np.outer(DeltaX, DeltaG))
            aH = np.dot(np.dot(V, aH), V.T) + rho * np.outer(DeltaX, DeltaX)
        return aH,tracing_list

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot(tracing1, tracing2):
    # 确保 tracing1 和 tracing2 是 numpy 数组
    import numpy as np
    tracing1 = np.array(tracing1)
    tracing2 = np.array(tracing2)

    # 创建一个新的图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 tracing1 的轨迹
    ax.plot(tracing1[:, 0], tracing1[:, 1], tracing1[:, 2], label='DFP')
    ax.scatter(*tracing1[0], color='red', label='Start')
    ax.scatter(*tracing1[-1], color='blue', label='End')

    # 绘制 tracing2 的轨迹
    ax.plot(tracing2[:, 0], tracing2[:, 1], tracing2[:, 2], label='BFGS', linestyle='--')

    # 添加图例
    ax.legend()

    # 添加标题和轴标签
    ax.set_title('Trajectory Plot')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.savefig('trajctary-DFP-BFGS-13.png')


if __name__ == "__main__":
    # print(func(np.array([0,0,0])))
    simulator = Simulator()
    H1,tracing1 = simulator.DFP_algorihtm(np.array([1,1,1]))
    H2,tracing2 = simulator.BFGS_algorithm(np.array([1,1,1]))
    # print(tracing1,'\n',tracing2)
    # plot(tracing1,tracing2)
    # print(H1)
    # print(H2)
    # print(np.dot(H1,H2))
    print(len(tracing1),len(tracing2))