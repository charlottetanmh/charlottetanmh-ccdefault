#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask


# In[ ]:


app = Flask(__name__)


# In[ ]:


from flask import render_template, request
import joblib

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        age = request.form.get("age")
        income = request.form.get("income")
        loan = request.form.get("loan") 
        age = float(age)
        income = float(income)
        loan = float(loan)
        print(age, income, loan)
        model1 = joblib.load("ccdefault_dt")
        pred1 = model1.predict([[age, income, loan]])
        s1 = "The Credit Card Default Score based on Decision Tree is " + str(pred1[0])
        model2 = joblib.load("ccdefault_lreg")
        pred2 = model2.predict([[age, income, loan]])
        s2 = "The Credit Card Default Score based on Regression is " + str(pred2[0])
        model3 = joblib.load("ccdefault_nn")
        pred3 = model3.predict([[age, income, loan]])
        s3 = "The Credit Card Default Score based on Neural Network is " + str(pred3[0])
        model4 = joblib.load("ccdefault_gbc")
        pred4 = model4.predict([[age, income, loan]])
        s4 = "The Credit Card Default Score based on Gradient Boosting is " + str(pred4[0])
        model5 = joblib.load("ccdefault_rfc")
        pred5 = model5.predict([[age, income, loan]])
        s5 = "The Credit Card Default Score based on Random Forest is " + str(pred5[0])
        return(render_template("index.html", result1 = s1, result2 = s2, result3 = s3, result4 = s4, result5 = s5))
    else:
        return(render_template("index.html", result1 = "2", result2 = "2", result3 = "2", result4 = "2", result5 = "2"))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:





# In[ ]:




