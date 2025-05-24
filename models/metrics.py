import mlx.core as mx

def mse(prediction: mx.array, reference: mx.array) -> mx.array:
    """
    Mean squared error
    """
    return mx.mean((prediction-reference)**2)

def rmse(prediction: mx.array, reference: mx.array) -> mx.array:
    return mx.sqrt(mse(prediction=prediction, reference=reference))

def l2_error(prediction: mx.array, reference: mx.array, eps: float = 1e-7) -> mx.array:
    """
    L2 error
    """
    squared_error = (prediction - reference)**2
    return mx.sqrt( squared_error.sum() )

def l2_relative_error(prediction: mx.array, reference: mx.array, eps: float = 1e-7) -> mx.array:
    """
    L2 relative error
    """
    squared_ref = reference**2
    return l2_error(prediction=prediction, reference=reference) / (mx.sqrt(squared_ref.sum()) + eps)