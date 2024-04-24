import numpy as np

class Simulator:
    def __init__(self , alpha = 0.1 , beta = 0.8 , ita = 1e-3):
        self.m = 5
        self.n = 4
        self.ita = ita
        self.alpha = alpha
        self.beta = beta
        np.random.seed(0)
        self.a = np.random.rand(self.m, self.n)

    def function(self,x):
        result = 0
        for i in range(0, self.m):
            result = result + np.exp( np.dot(self.a[i],x) ) + np.exp( np.dot(-self.a[i],x) )
        return result

    def differential_evolution(self,x):
        result = np.zeros(self.n)
        for i in range(0, self.m):
            result = result + self.a[i] * np.exp( np.dot(self.a[i],x) ) - self.a[i] * np.exp( np.dot(-self.a[i],x) )
        return result

    def backtracking_line_search(self,x,direction):
        t = 1
        differential = self.differential_evolution(x)
        while self.function(x + t * direction) > self.function(x) + self.alpha * t * np.dot(differential, direction):
            t = self.beta * t
        return t 


    def steepst_direction(self,gradient):
        result = np.zeros(self.n)
        result[gradient >= 0] = -1
        result[gradient < 0] = 1
        result = result * np.argmax(np.abs(gradient))
        return result

    def gradient_descent(self , tracing = False, length = False):
        #x = 1 initial point
        x = np.array([1 for i in range(0, self.n)])
        tracing_list = []
        step = 0
        if tracing:
            P_ = self.function(np.zeros(self.n))
        while True:
            gradient = self.differential_evolution(x)
            direction = self.steepst_direction(gradient)
            if np.linalg.norm(gradient) < self.ita:
                break
            t = self.backtracking_line_search(x, direction)
            x = x + t * direction
            step = step + 1
            if tracing:
                tracing_list.append(np.log(self.function(x) - P_))
        return x , tracing_list , step

def plot(tracing_list):
    import matplotlib.pyplot as plt
    plt.plot(tracing_list)
    plt.xlabel('step')
    plt.ylabel('log(f(x) - f(x*))')
    #save
    plt.savefig('log(f(x) - f(x*)).png')

if __name__ == "__main__":
    # simulator = Simulator()
    # x , tracing_list , step = simulator.gradient_descent(tracing = True)
    # plot(tracing_list)
    Ans = np.zeros((9,9))
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for beta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            simulator = Simulator(alpha = alpha, beta = beta)
            x , _ , step = simulator.gradient_descent()
            Ans[int(alpha*10)-1][int(beta*10)-1] = step
    print(Ans)
    #draw Ans to a heatmap
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.heatmap(Ans, annot = True)
    plt.xlabel('beta')
    plt.ylabel('alpha')
    plt.savefig('heatmap.png')
    