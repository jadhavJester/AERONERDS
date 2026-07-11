import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def linear_func(x, a, b):
    return a * x + b

def exp_func(x, a, b):
    # H = a * exp(b * x)
    return a * np.exp(b * x)

class DegradationTrend:
    def __init__(self, maintenance_threshold=0.75):
        self.maintenance_threshold = maintenance_threshold
        self.popt = None
        self.func_type = None
        
    def fit(self, cycles, health_values):
        """Fits both linear and exp trends, picks the best by R^2."""
        if len(cycles) < 2:
            return False
            
        cycles = np.array(cycles)
        health_values = np.array(health_values)
        
        # Linear fit
        try:
            popt_lin, _ = curve_fit(linear_func, cycles, health_values)
            y_pred_lin = linear_func(cycles, *popt_lin)
            r2_lin = r2_score(health_values, y_pred_lin)
        except:
            r2_lin = -np.inf
            popt_lin = None
            
        # Exponential fit
        try:
            # Initial guess: a ~ initial health, b ~ negative slope
            p0_exp = [health_values[0], -0.001]
            popt_exp, _ = curve_fit(exp_func, cycles, health_values, p0=p0_exp, maxfev=2000)
            y_pred_exp = exp_func(cycles, *popt_exp)
            r2_exp = r2_score(health_values, y_pred_exp)
        except:
            r2_exp = -np.inf
            popt_exp = None
            
        if r2_lin == -np.inf and r2_exp == -np.inf:
            return False
            
        if r2_exp > r2_lin:
            self.func_type = 'exponential'
            self.popt = popt_exp
        else:
            self.func_type = 'linear'
            self.popt = popt_lin
            
        return True
        
    def predict_health(self, cycles):
        """Predicts health for given cycles based on fitted trend."""
        if self.func_type == 'linear':
            return linear_func(np.array(cycles), *self.popt)
        elif self.func_type == 'exponential':
            return exp_func(np.array(cycles), *self.popt)
        return None
        
    def estimate_rul_crossover(self):
        """Estimates the cycle at which health crosses the maintenance threshold."""
        if self.func_type is None:
            return None
            
        if self.func_type == 'linear':
            a, b = self.popt
            if a >= 0:
                # Not degrading
                return np.inf
            # a*c + b = threshold => c = (threshold - b) / a
            return (self.maintenance_threshold - b) / a
            
        elif self.func_type == 'exponential':
            a, b = self.popt
            if b >= 0 or a <= 0:
                return np.inf
            # a*exp(b*c) = threshold => c = ln(threshold/a) / b
            # Ensure argument to log is positive
            val = self.maintenance_threshold / a
            if val <= 0:
                return np.inf
            return np.log(val) / b
            
        return None

if __name__ == "__main__":
    # Test script
    cycles = np.arange(1, 31)
    health = 1.0 - 0.005 * cycles + np.random.normal(0, 0.01, 30)
    
    dt = DegradationTrend(0.75)
    dt.fit(cycles, health)
    print(f"Fit Type: {dt.func_type}")
    print(f"Params: {dt.popt}")
    print(f"Crosses 0.75 at cycle: {dt.estimate_rul_crossover():.1f}")
