from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PLANT_", env_file=".env", extra="ignore")

    data_dir: str = "./datasets/processed"
    model_file: str = "experiments/efficientnet_b0/best_model.pth"
    model_name: str = "efficientnet_b0"
    device: str = "cpu"  # or "cuda" for GPU    

settings = Settings()
