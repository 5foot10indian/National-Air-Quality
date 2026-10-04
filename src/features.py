df["Month"] = df["Date"].dt.month
def get_season(month):
    if month in [12,1,2]:
        return "Winter"
    elif month in [3,4,5]:
        return "Summer"
    elif month in [6,7,8,9]:
        return "Monsoon"    
    else:
        return "Post-Monsoon"

df["Season"] = df["Month"].apply(get_season)  