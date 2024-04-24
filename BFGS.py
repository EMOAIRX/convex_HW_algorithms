import numpy as np

class NormalFunction:
    def __init__(self):
        pass

    def __call__(self,x):
        return (x[0]**4)/4+(x[1]**2)/2-x[0]*x[1]+x[0]-x[1]

    def gradient(self,x):
        return np.array([x[0]**3 - x[1] + 1, x[1] - x[0] - 1])

class Simulator:
    def __init__(self ,ita = 1e-8):
        self.n = 100
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
        aH = np.eye(2)
        tracing_list = []
        x = start_point
        grad = self.f_grad(x)
        step = 0
        while np.linalg.norm(grad) >= self.ita:
            direction = -np.dot(aH, grad)
            alpha = self.exact_line_search(x,direction)
            x = x + alpha * direction
            tracing_list.append(x)
            grad_old = grad
            grad = self.f_grad(x)
            DeltaX = alpha * direction
            DeltaG = grad - grad_old
            #DFP algorithm
            aH = aH + np.outer(DeltaX, DeltaX) / np.dot(DeltaX, DeltaG) - np.outer(np.dot(aH, DeltaG),np.dot(aH, DeltaG)) / np.dot(np.dot(DeltaG, aH), DeltaG)        

        return x,tracing_list
    
    def BFGS_algorithm(self, start_point):


import matplotlib.pyplot as plt
def plot(tracing1, tracing2):
    # 确保 tracing1 和 tracing2 是 numpy 数组
    import numpy as np
    tracing1 = np.array(tracing1)
    tracing2 = np.array(tracing2)

    # 创建一个新的图
    plt.figure()

    # 绘制 tracing1 的轨迹
    plt.plot(tracing1[:, 0], tracing1[:, 1], label='Trajectory 1')
    plt.scatter(*tracing1[0], color='red', label='Start of Trajectory 1')
    plt.scatter(*tracing1[-1], color='blue', label='End of Trajectory 1')
    plt.plot(tracing2[:, 0], tracing2[:, 1], label='Trajectory 2', linestyle='--')
    plt.scatter(*tracing2[0], color='red', label='Start of Trajectory 2')
    plt.scatter(*tracing2[-1], color='blue', label='End of Trajectory 2')
    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title('Trajectory Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 显示网格
    plt.grid(True)

    plt.savefig('trajctary-13.png')



if __name__ == "__main__":
    simulator = Simulator()
    _,tracing1 = simulator.DFP_algorihtm(np.array([0,0]))
    _,tracing2 = simulator.DFP_algorihtm(np.array([1.5,1]))
    print(tracing1,'\n',tracing2)
    plot(tracing1,tracing2)
    