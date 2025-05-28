"""
This module contains samplers that generate points at which to enforce physics, initial condition, and boundary condition constraints.
These points can then be fed into the appropriate loss.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from configs.schema import BatchSizes
import mlx.core as mx

class Sampler(ABC):

    def __init__(self,
                 batch_sizes: BatchSizes,
                 t_lims: List,
                 x_lims: List,
                 y_lims: Optional[List] = None):
        self.batch_sizes = batch_sizes
        self.t_lims = t_lims
        self.x_lims = x_lims
        self.y_lims = y_lims 

        if y_lims:
            self._diff = mx.array(
                [
                    t_lims[1] - t_lims[0],
                    x_lims[1] - x_lims[0],
                    y_lims[1] - y_lims[0],
                ])
            self._b = mx.array([t_lims[0], x_lims[0], y_lims[0]])
        else:
            self._diff = mx.array(
                [
                    t_lims[1] - t_lims[0],
                    x_lims[1] - x_lims[0],
                ])
            self._b = mx.array([t_lims[0], x_lims[0]])

    @abstractmethod
    def sampler(self, shape: tuple[int, int]) -> mx.array:
        """
        Sampler returns an mx.array of shape (n_samples, n_dims),
        with each entry in [0, 1].
        """
        pass

    def _solution(self) -> tuple[mx.array, mx.array]:
        batch_size = self.batch_sizes.solution
        if self.y_lims:
            samples = self.sampler(shape=(batch_size, 3))
        else:
            samples = self.sampler(shape=(batch_size, 2))
        batch = self._b + self._diff * samples

        return batch[:,1:], batch[:, [0]]
    
    def _initial(self) -> tuple[mx.array, mx.array]:
        batch_size = self.batch_sizes.initial
        t = mx.ones((batch_size,1)) * self.t_lims[0]
        if self.y_lims:
            samples = self.sampler(shape=(batch_size, 2))
        else:
            samples = self.sampler(shape=(batch_size, 1))
        batch = self._b[1:] + self._diff[1:] * samples

        return batch, t

    def _sample_perimeter(self, n_samples: int, x_lims: List, y_lims: Optional[List] = None):
        if y_lims:
            x_width = x_lims[1] - x_lims[0]
            y_width = y_lims[1] - y_lims[0]

            perimeter = 2 * x_width + 2 * y_width
            s = perimeter * mx.random.uniform(shape=(n_samples,)) 

            # Boolean masks for each edge
            bottom_mask = s < x_width
            right_mask = x_width <= s < x_width + y_width
            top_mask = x_width + y_width <= s <  2*x_width + y_width
            left_mask = s >= 2*x_width + y_width

            samples = mx.zeros(shape=(n_samples, 2))
            
            samples[bottom_mask, 0] = x_lims[0] + x_width * s[bottom_mask]
            samples[bottom_mask, 1] = y_lims[0]
            samples[right_mask, 0] = x_lims[1] 
            samples[right_mask, 1] = y_lims[0] + y_width * (s[right_mask] - x_width)
            samples[top_mask, 0] = x_lims[0] + x_width * (s[bottom_mask] - x_width - y_width)
            samples[top_mask, 1] = y_lims[1]
            samples[left_mask, 0] = x_lims[0] 
            samples[left_mask, 1] = y_lims[0] + y_width * (s[right_mask] - 2*x_width - y_width)

            return samples
        else:
            # in the 1d case, return either endpoint
            samples = mx.random.bernoulli(p=0.5, shape=(n_samples, 1))
            return x_lims[0] + (x_lims[1] - x_lims[0]) * samples

    def _boundary(self) -> tuple[mx.array, mx.array]:
        batch_size = self.batch_sizes.boundary
        t = self._b[0] + self._diff[0] * self.sampler(shape=(batch_size, 1))
        batch = self._sample_perimeter(batch_size, self.x_lims, self.y_lims)
        return batch, t
    
    def get_batch(self) -> dict[str, tuple[mx.array, mx.array]]:
        return {
            k: eval(f"self._{k}()") for k in ['solution', 'initial', 'boundary'] 
        }

class UniformSampler(Sampler):
    """
    Uniform Sampler
    """
    def sampler(self, shape: tuple[int, int]) -> mx.array:
        return mx.random.uniform(shape=shape)
    
class LHSampler(Sampler):
    """
    Latin Hypercube Sampler
    """
    def sampler(self, shape: tuple[int, int]) -> mx.array:
        raise NotImplementedError("LHS not implemented yet")
    
class SobolSampler(Sampler):
    """
    Sobol Sequence Sampler
    """
    def sampler(self, shape: tuple[int, int]) -> mx.array:
        raise NotImplementedError("Sobol not implemented yet")

