# %%
import research.start  # noqa

# %%
import pandas as pd

from models.model import Model
from pento.data_processing import process_data


df = pd.read_csv("data/x.csv")
df = process_data(df)

model = Model()
