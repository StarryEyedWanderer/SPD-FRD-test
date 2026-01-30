from pydantic import BaseModel, Field


class SimConfig(BaseModel):
    dt: float = 0.01
    T: int = 2000
    mu0: float = -1.0
    mu1: float = 1.0
    omega: float = 2.0
    sigma: float = 0.1
    seed: int = 1


class FitConfig(BaseModel):
    window_size: int = 500
    model_type: str = "linear2d"


class ThresholdConfig(BaseModel):
    baseline_fraction: float = Field(0.25, ge=0.01, le=0.9)
    quantile: float = Field(0.99, ge=0.5, le=0.999)


class EventConfig(BaseModel):
    rolling_window: int = 200
    k_sigma: float = 5.0
