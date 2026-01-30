from pydantic import BaseModel


class RunMeta(BaseModel):
    dt: float
    T: int
    mu0: float
    mu1: float
    omega: float
    sigma: float
    seed: int
