import numpy as np

class Rosenbrock:
    def __init__(self,n = 100,alpha = 1):
        self.alpha = alpha
        self.n = n

    def __call__(self,x):
        sum = 0
        for i in range(0,self.n // 2):
            sum += self.alpha * (x[2*i+1] - x[2*i] ** 2) ** 2 + (1 - x[2*i]) ** 2
        return sum

    def gradient(self,x):
        grad = np.zeros(self.n)
        for i in range(0,self.n // 2):
            grad[2*i] = -4 * self.alpha * x[2*i] * (x[2*i+1] - x[2*i] ** 2) - 2 * (1 - x[2*i])
            grad[2*i+1] = 2 * self.alpha * (x[2*i+1] - x[2*i] ** 2)
        return grad

class Simulator:
    def __init__(self ,ita = 1e-9):
        self.n = 100
        self.function = Rosenbrock(n = self.n,alpha = 1)
        self.ita = ita
        self.btls_alpha = 0.3 #backtracing_line_search
        self.btls_beta = 0.9


    def backtracking_line_search(self,x,direction):
        t = 1
        differential = self.f_grad(x)
        while self.f(x + t * direction) > self.f(x) + self.btls_alpha * t * np.dot(differential, direction):
            t = self.btls_beta * t
        return t 

    def f(self,x):
        return self.function(x)

    def f_grad(self,x):
        return self.function.gradient(x)

    def Hestenes_Stiefel(x,direction,grad,grad_old):
        #print(np.dot(grad,grad - grad_old),'.',np.dot(direction,grad-grad_old))
        return np.dot(grad,grad - grad_old) / np.dot(direction,grad - grad_old)
    
    def Polak_Ribiere(x,direction,grad,grad_old):
        return np.dot(grad,grad - grad_old) / np.dot(grad_old,grad_old)
    
    def Fletcher_Beeves(x,direction,grad,grad_old):
        #print(np.dot(grad,grad),'.',np.dot(grad_old,grad_old))
        return np.dot(grad,grad) / np.dot(grad_old,grad_old)

    def conjugate_gradient(self, start_point, optimal_point, BetaMethod):

        tracing_list = []

        optimal_value = self.f(optimal_point)
        x = start_point
        grad = self.f_grad(x)
        direction = -grad
        
        tracing_list.append(np.log(self.f(x) - optimal_value))

        step = 0
        while np.linalg.norm(grad) >= self.ita:
            alpha = self.backtracking_line_search(x,direction)
            x = x + alpha * direction
            #print('A',np.dot(direction, np.dot(Qmatrix, direction)))
            
            tracing_list.append(np.log(self.f(x) - optimal_value))
            #print(self.f(x) - optimal_value)

            grad_old = grad
            grad = self.f_grad(x)

            #print(np.linalg.norm(grad))

            beta = BetaMethod(direction,grad,grad_old)
            direction = -grad + beta * direction
#            if step == 100 : break
            step += 1
        
        
        return x,tracing_list
            


        

def plot(tracing_list1,tracing_list2,tracing_list3):
    import matplotlib.pyplot as plt
    #red as tracing1 and "Hestenes_Stiefel" as label
    plt.plot(tracing_list1,'r',label = 'Hestenes_Stiefel')
    #blue as
    plt.plot(tracing_list2,'b',label = 'Polak_Ribiere')
    #Fletcher_Beeves
    plt.plot(tracing_list3,'g',label = 'Fletcher_Beeves')
    #output label 
    plt.legend()
    plt.title("alpha = 1")  # Add this line to set the title of the plot

    plt.savefig('log(f(x) - f(x*))-alpha=1.png')

    

if __name__ == "__main__":
    simulator = Simulator()
    _,tracing1 = simulator.conjugate_gradient(
        np.array([-1 for i in range(100)]),np.array([1 for i in range(100)])
        ,simulator.Hestenes_Stiefel
    )
    _,tracing2 = simulator.conjugate_gradient(
        np.array([-1 for i in range(100)]),np.array([1 for i in range(100)])
        ,simulator.Polak_Ribiere
    )
    _,tracing3 = simulator.conjugate_gradient(
        np.array([-1 for i in range(100)]),np.array([1 for i in range(100)])
        ,simulator.Fletcher_Beeves
    )
    plot(tracing1,tracing2,tracing3)
    