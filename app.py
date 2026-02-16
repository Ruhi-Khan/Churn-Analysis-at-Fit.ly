from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import os

app = Flask(__name__)

# Create static folder
if not os.path.exists("static"):
    os.makedirs("static")

# ------------------ DATA GENERATION ------------------
def generate_data():
    np.random.seed(42)
    data = []

    for i in range(150):
        age = np.random.randint(18, 60)
        subscription = np.random.choice(['Silver','Gold','Platinum'])

        if subscription == 'Silver':
            fee = 500
            usage = np.random.randint(0,10)

        elif subscription == 'Gold':
            fee = 1000
            usage = np.random.randint(0,12)

        else:
            fee = 2000
            usage = np.random.randint(0,15)

        # ----------- CHURN LOGIC BASED ON USAGE ----------
        if usage <= 1:
            churn = 1  # almost sure churn
        elif usage <= 3:
            churn = np.random.choice([0,1], p=[0.3,0.7])
        elif usage <= 8:
            churn = np.random.choice([0,1], p=[0.85,0.15])
        else:
            churn = np.random.choice([0,1], p=[0.95,0.05])

        data.append([i, age, subscription, fee, usage, churn])

    df = pd.DataFrame(data, columns=[
        "CustomerID","Age","SubscriptionType",
        "MonthlyFee","UsageHours","Churn"
    ])
    return df

# ------------------ MAIN ROUTE ------------------
@app.route("/", methods=["GET","POST"])
def home():

    df = generate_data()

    # Convert subscription to numbers
    df_model = df.copy()
    df_model["SubscriptionType"] = df_model["SubscriptionType"].map({
        "Silver":1, "Gold":2, "Platinum":3
    })

    X = df_model[["Age","SubscriptionType","MonthlyFee","UsageHours"]]
    y = df_model["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.3,random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)

    # ------------------ PLOTS ------------------
    plt.style.use("ggplot")

    # Bar chart
    churn_counts = df.groupby("SubscriptionType")["Churn"].sum()
    plt.figure()
    churn_counts.plot(kind="bar")
    plt.title("Churn by Subscription Type")
    plt.savefig("static/bar.png")
    plt.close()

    # Pie chart
    churn_total = df["Churn"].sum()
    stay_total = len(df) - churn_total

    plt.figure()
    plt.pie([stay_total,churn_total], labels=["Stayed","Churned"], autopct="%1.1f%%")
    plt.title("Overall Churn Percentage")
    plt.savefig("static/pie.png")
    plt.close()

    # Confusion Matrix
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.savefig("static/confusion.png")
    plt.close()

    prediction_text = ""

    # ------------------ USER PREDICTION ------------------
    if request.method == "POST":
        age = int(request.form["age"])
        subscription = int(request.form["subscription"])
        usage = int(request.form["usage"])

        # Prevent negative usage
        if usage < 0:
            usage = 0

        if subscription == 1:
            fee = 500
        elif subscription == 2:
            fee = 1000
        else:
            fee = 2000

        new_data = [[age, subscription, fee, usage]]
        result = model.predict(new_data)[0]

        # ---------- BUSINESS RULES OVERRIDE ----------
        if usage <= 1:
            prediction_text = "🔴 VERY LIKELY TO CHURN (Not using app)"
        elif usage <= 3:
            prediction_text = "🟠 MAY CHURN (Low usage)"
        elif usage <= 8:
            prediction_text = "🟢 Likely to STAY (Active user)"
        else:
            prediction_text = "🔥 Power User - Very Likely to STAY"

    total_customers = len(df)
    total_churn = df["Churn"].sum()
    churn_percent = round((total_churn/total_customers)*100,2)

    return render_template("index.html",
                           total_customers=total_customers,
                           total_churn=total_churn,
                           churn_percent=churn_percent,
                           accuracy=round(accuracy*100,2),
                           prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=False, port=8000)