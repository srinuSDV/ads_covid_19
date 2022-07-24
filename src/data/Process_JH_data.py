import pandas as pd
import numpy as np

from datetime import datetime

def store_relational_JH_data():
    data_path_1 = r"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/" \
                  r"master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    pd_raw = pd.read_csv(data_path_1)
    pd_data_base = pd_raw.rename(columns={"Country/Region": "country", "Province/State": "state"})
    pd_data_base["state"] = pd_data_base["state"].fillna("no")
    pd_data_base = pd_data_base.drop(["Lat", "Long"], axis=1)
    pd_relational_model = pd_data_base.set_index(['state', 'country']) \
        .T \
        .stack(level=[0, 1]) \
        .reset_index() \
        .rename(columns={'level_0': 'date',
                         0: 'confirmed'},
                )
    pd_relational_model["date"] = pd_relational_model.date.astype("datetime64[ns]")
    pd_relational_model.confirmed = pd_relational_model.confirmed.astype(int)
    pd_relational_model.to_csv('../data/processed/COVID_relational_confirmed.csv', sep=';', index=False)
    print("No. of rows stored:" +str(pd_relational_model.shape[0]))
    print("Data Acquired and Stored as COVID_relational_confirmed.csv")


def store_flat_table_JH_data():

    data_path_1 = r"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/" \
                  r"master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    pd_raw = pd.read_csv(data_path_1)
    time_idx = pd_raw.columns[4:]
    df_plot = pd.DataFrame({"date": time_idx})
    Country_list = ["Italy", "India", "US", "Germany"]
    for each in Country_list:
        df_plot[each] = np.array(pd_raw[pd_raw["Country/Region"] == each].iloc[:, 4::].sum(axis=0))

    time_idx = [datetime.strptime(each, "%m/%d/%y") for each in df_plot.date]
    time_str = [each.strftime('%Y-%m-%d') for each in time_idx]
    df_plot["date"] = time_idx
    df_plot.to_csv('../data/processed/COVID_small_flat_table.csv', sep=';', index=False)
    print("Data Acquired and Stored as COVID_small_flat_table.csv")




if __name__ == "__main__":
    print("Preparing data for Confirmed Covid Cases and Doubling rate Ploting... ")
    store_relational_JH_data()
    print("Now preparing data for SIR Modelling and Ploting... ")
    store_flat_table_JH_data()