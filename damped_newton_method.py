import numpy as np

class Simulator:
    def __init__(self , alpha = 0.3 , beta = 0.9 , ita = 1e-9):
        self.n = 1000
        self.ita = ita
        self.alpha = alpha
        self.beta = beta
        #generate Q = 1000 * 1000, condition number >=100
        self.Q = np.random.rand(1000,1000)
        self.Q = np.dot(self.Q.T,self.Q)
        self.Q = self.Q + 100 * np.eye(1000)
        self.b = np.random.rand(1000)

    def f(self,x):
        return 0.5 * np.dot(np.dot(x.T,self.Q),x) - np.dot(self.b.T,x)

    def f_grad(self,x):
        return np.dot(self.Q,x) - self.b
    
    def f_hess(self,x):
        return self.Q


    def backtracking_line_search(self,x,direction):
        t = 1
        differential = self.f_grad(x)
        while self.f(x + t * direction) > self.f(x) + self.alpha * t * np.dot(differential, direction):
            t = self.beta * t
        return t 
    
    def exact_line_search(self,x,direction):
        # argmin (1/2(x+direction*t)^T Q (x+direction*t) - b^T (x+direction*t))
        # Q(x+t*direction) direction - b = 0
        # t = (b - Qx) / (Qdirection)
        t = -np.dot(self.f_grad(x),direction) / np.dot(np.dot(direction.T,self.Q),direction)
        return t

    def damped_newton_method(self , tracing = False, length = False, P_ = None):
        #x = 1 initial point
        x = np.array([np.random.randint(-100,100) for i in range(1000)])
        tracing_list = []
        step = 0
        # if tracing:
            # P_ = self.f(np.zeros(self.n))

        while True:
            grad = self.f_grad(x)
            if np.linalg.norm(grad) < self.ita:
                print('grad_norm:', np.linalg.norm(grad))
                break
            hess = self.f_hess(x)
            inv_hess = np.linalg.inv(hess)
            direction = -np.dot(inv_hess, grad)
            t = self.backtracking_line_search(x, direction) 
            print(t)
            x = x + t * direction
            step = step + 1
            if tracing:
                tracing_list.append(np.log(self.f(x) - P_))
            # print()
            print('step:', step, 'f(x):', self.f(x))
        return x , tracing_list , step

    def gradient_descent(self, P_):
        x = np.array([np.random.randint(-100,100) for i in range(1000)])
        tracing_list = []
        step = 0

        while True:
            grad = self.f_grad(x)
            if np.linalg.norm(grad) < self.ita:
                break
            direction = -grad
            t = self.exact_line_search(x, direction) 
            x = x + t * direction
            step = step + 1
            tracing_list.append(np.log(self.f(x) - P_))
            print('step:', step,'f(x):', self.f(x))
        return x , tracing_list , step
        

def plot(tracing_list):
    import matplotlib.pyplot as plt
    print(tracing_list)
    plt.plot(range(len(tracing_list)),tracing_list)
    plt.xlabel('step')
    plt.ylabel('log(f(x) - f(x*))')
    #save
    plt.savefig('log(f(x) - f(x*))-step-conjugate.png')

if __name__ == "__main__":
    simulator = Simulator()
    # x1,tracing_list1,step1 = simulator.damped_newton_method(tracing = False)

    # min 0.5 * np.dot(np.dot(x.T,self.Q),x) - np.dot(self.b.T,x)
    argminx = np.dot(np.linalg.inv(simulator.Q),simulator.b)
    P_ = simulator.f(argminx)

    # P_ = simulator.f(x1)
    x1,tracing_list2,step1 = simulator.damped_newton_method(tracing = True , P_ = P_)
    # x2,tracing_list2,step2 = simulator.gradient_descent(P_ = P_)
    plot(tracing_list2)
    