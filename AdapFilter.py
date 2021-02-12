import matplotlib.pyplot as plt
import numpy as np
from numpy import exp

class Simulator():
    
    def __init__(self, intensities, change_points):
        
        self.intensity = intensities
        self.change_points = change_points
        self.events = []
        self.end_time = change_points[-1]
    
    def simulate(self):
        
        self.events = []
        
        for i, time in enumerate(self.change_points):
            if i == 0:
                time_interval = time
            else:
                time_interval = time - self.change_points[i-1]
            
            intensity = self.intensity[i]
            
            N_events = np.random.poisson(intensity * time_interval)
            
            events = np.sort(np.random.random(N_events))
            events = (time - time_interval) + time_interval*events
            
            self.events.extend(events)
    
    def __call__(self):
        self.simulate()
        return self.events
    
    def plot_process(self):
        x = [0,]+self.events
        y = np.arange(len(self.events)+1)-1
        plt.step(x, y,
                 where='post', label='post')
        if len(x)<20:
            plt.plot(x[1:], y[1:], 'C1o', alpha=0.5)
            
    def plot_intensity(self):
        x = [0,] + list(self.change_points)
        y = list(self.intensity) + [self.intensity[-1]]
        plt.step(x, y, where = 'post', c = 'black')
            
    def __repr__(self):
        rep = f'''
        Intensities \t\t: {self.intensity}
        Change-points \t\t: {self.change_points}
        Number of events\t: {len(self.events)}
        '''
        return rep

class Filter():
    
    def __init__(self, learning_rate = 0):
        
        self.learning_rate = learning_rate
        
    @staticmethod
    def merge_sorted_arrays(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        m,n = len(a), len(b)
        # Get searchsorted indices
        idx = np.searchsorted(a,b)
    
        # Offset each searchsorted indices with ranged array to get new positions
        # of b in output array
        b_pos = np.arange(n) + idx
    
        l = m+n
        mask = np.ones(l,dtype=bool)
        out = np.empty(l,dtype=np.result_type(a,b))
        mask[b_pos] = False
        out[b_pos] = b
        out[mask] = a
        
        return out, mask + 0
    
    @staticmethod
    def approx_dash(x):
        return - x/6 + x**2/24 - x**3/120 + x**4/720 - x**5/5040
    
    @staticmethod
    def approx_eta(x):
        return 1 - x/2 - x**2/6 + x**3/24 - x**4/120 + x**5/720
    
    def estimation_single_step(self, delta_N, delta_t, N, T, N_dash, T_dash, lambda_, lambda_dash, J_dash, eta):
        
        J_dash = lambda_dash * (delta_t - delta_N/max(1e-20, lambda_))
        eta = max(0, eta - self.learning_rate * J_dash)
        
        N_dash = - delta_t * (delta_t - delta_N/max(1e-20, lambda_))
        N = exp(-eta*delta_t) * N + delta_N
        
        c = delta_t * eta
        
        if c < 1e-7:
            T_dash = - delta_t * exp(-c) * T - delta_t**2/2 + (c+1)*delta_t**2 * self.approx_dash(c)
            T = exp(-c) * T + delta_t * self.approx_eta(c)
        elif c < 1e-5:
            T_dash = - delta_t * exp(-c) * T - delta_t**2/2 + (c+1)*delta_t**2 * self.approx_dash(c)
            T = exp(-c) * T + (1 - exp(-c))/eta
        else:
            T_dash = - delta_t * exp(-c) * T - (1 - (1 + c) * exp(-c))/(eta**2)
            T = exp(-c) * T + (1 - exp(-c))/eta
            
        lambda_dash = (N_dash * T - N * T_dash) / (T**2)
        lambda_ = N / T
        
        return N, T, N_dash, T_dash, lambda_, lambda_dash, J_dash, eta
        
    
    def estimate(self, end_time = None, N_reads = 100):
        
        # Values we might wish to store:
        self.N = np.zeros(N_reads)
        self.T = np.zeros(N_reads)
        self.lambdas = np.zeros(N_reads)
        self.eta = np.zeros(N_reads)
        j = 0 # index for storing
        
        # Values we need to store during the estimation phase.
        N = 0
        T = 0
        N_dash = 0
        T_dash = 0
        lambda_ = 0
        lambda_dash = 0
        J_dash = 0
        eta = 0
        
        # Use the end time for the process (either give or inferred) to choose the update times.
        if end_time == None:
            end_time = np.ceil(self.events[-1])
        
        self.estimation_times = np.linspace(0, end_time, N_reads+1)[1:]
        
        update_times, event_occurrence = self.merge_sorted_arrays(self.events, self.estimation_times)
        
        for i, (t, delta_N) in enumerate(zip(update_times, event_occurrence)):
            
            if i == 0:
                delta_t = t
            else:
                delta_t = t - update_times[i-1]
            
            N, T, N_dash, T_dash, lambda_, lambda_dash, J_dash, eta = self.estimation_single_step(
                delta_N, delta_t, N, T, N_dash, T_dash, lambda_, lambda_dash, J_dash, eta
            )
            
            if delta_N == 0:
                # if this is an update time, i.e. not an event time, log any useful values as class attributes.
                self.N[j] = N
                self.T[j] = T
                self.lambdas[j] = lambda_
                self.eta[j] = eta
                j += 1
        
    def __call__(self, process, N_reads = 100):
        
        self.process = process
        
        if isinstance(process, Simulator):
            self.events = process()
            self.estimate(end_time = process.end_time, N_reads = N_reads)
        else:
            self.events = process
            self.estimate(N_reads = N_reads)
        
        return self.lambdas
    
    def plot(self):
        
        plt.plot(self.estimation_times, self.lambdas, linewidth = 1)
        
        if isinstance(self.process, Simulator):
            self.process.plot_intensity()


intensities = [1, 2, 1, 5]
change_points = [3000, 6000, 15000, 20000]

poisson_process = Simulator(intensities, change_points)
poisson_process()
#poisson_process.plot_process()
adap_filter = Filter(learning_rate = 0.0001)
adap_filter(poisson_process, N_reads = 1000)
adap_filter.plot()
