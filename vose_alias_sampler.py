'''
Created on Aug 9, 2016

Samples from discrete distribution in O(1)

See: http://www.keithschwarz.com/darts-dice-coins/

@author: mkroutikov
'''
import random

class VoseAliasSampler:
    
    def __init__(self, weights):
        '''Initializes sampler from the given list of (non-negative) weights. Weights are just unnormalized probabilities'''
        self._N = len(weights)

        assert all(w>0 for w in weights)
        tot = sum(weights)
        if tot == 0.0:
            raise RuntimeError('Bad weights: total probability is zero!?')
        weights = [w * len(weights) / tot for w in weights]
        
        small = [idx for idx,w in enumerate(weights) if w < 1.]
        large = [idx for idx,w in enumerate(weights) if w >= 1.]
        
        alias = [0.] * self._N
        prob  = [0.] * self._N
        while small and large:
            l = small.pop()
            g = large.pop()
            
            alias[l] = g
            prob[l] = weights[l]
            
            weights[g] -= 1. - weights[l]
            if weights[g] < 1.:
                small.append(g)
            else:
                large.append(g)
        
        while large:
            g = large.pop()
            prob[g] = 1.
            alias[g] = g
        
        while small:
            l = small.pop()
            prob[l] = 1.
            alias[l] = l
        
        self._prob = prob
        self._alias = alias
    
    def __call__(self):
        n = random.randint(0, self._N-1)
        
        if random.random() < self._prob[n]:
            return n
        
        return self._alias[n]


if __name__ == '__main__':
    
    sampler = VoseAliasSampler([0.25, 0.4, 0.35])
    
    import collections
    stats = collections.defaultdict(int)
    
    for _ in range(100000):
        stats[sampler()] += 1
    
    print(stats)
