import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math


def main(verbosity=False):
    st.header("Building a Regression Model with a more robust loss function")
    st.markdown("""
    In this script, a new loss function is implemented to the algorithm 
    that was used in the first wedo session in order to optimize the error punishment.

    """)

    st.header("Dataset")
    cal_housing = fetch_california_housing()
    X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target

    if verbosity:
        st.dataframe(X)

    df = pd.DataFrame(
        dict(MedInc=X['MedInc'], Price=y))

    st.write(df)

    # train test
    X_train, X_test, y_train, y_test = train_test_split(df.MedInc, df.Price, test_size=0.2, random_state=42)

    fig = px.scatter(df, x="MedInc", y="Price", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("General Model Prediction on Test Data")
    beta = reg(X_train, y_train, verbose=verbosity)
    st.latex(fr"Price = {beta[1]:.4f} \times MedInc + {beta[0]:.4f}")

    pred_list = []
    for i in X_test:
        y_pred = i * beta[1] + beta[0]
        pred_list.append(y_pred)
    df_pred = pd.DataFrame(dict(y=y_test, y_pred=pred_list))
    st.write(df_pred)

    err_list = []
    total_error = 0
    for i in range(df_pred.shape[0]):
        err_ = (df_pred['y'].iloc[i] - df_pred['y_pred'].iloc[i]) ** 2
        err_list.append(err_)
        total_error += err_
    temp = pd.DataFrame(dict(err=err_list, y_pred=df_pred['y_pred']))

    fig = px.scatter(temp, x="y_pred", y="err", trendline="ols", title="Loss Function vs. Prediction")
    st.plotly_chart(fig, use_container_width=True)
    mse = mean_squared_error(df_pred.y, df_pred.y_pred)
    mse = round(mse, 3)
    st.write("Mean Squared Error of General Model's predictions = " + str(mse))
    st.write("Total error = " + str(total_error))
    st.line_chart(df_pred[-300:])

    ################################################## NEW MODEL ####################

    st.subheader("Formulating the New Model")

    st.markdown("#### Prediction")
    st.latex(r"\hat{y}_i=\beta_0 + \beta_1 x_i")

    st.markdown("#### Loss Function")
    st.write('The generalized form of loss function can be written as: ')
    st.latex(
        r"f(x,\alpha,c) = \frac{\left\lvert \alpha-2 \right\rvert}{\alpha} (\frac{(\frac{x}{c})^2}{\left\lvert \alpha-2 \right\rvert}+1)^\frac{\alpha}{2} -1)")

    st.write("Here alpha is a parameter that controls the robustness of the loss and "
             "c > 0 is a scale parameter that controls the size of the loss's quadratic bowl near x=0")
    st.write("If alpha = 2, this function approaches to L2 loss (squared error) in the limit.")
    st.write("If alpha approaches to negative infinity, then the function becomes Welsch Loss.")

    st.write(
        "Welsch Loss Function is used in this assignment in order to get smoother and adaptive error calculations.")

    st.latex(r"L = \sum_{i=1}^{N}{(1 - e^{-\frac{1}{2}(y_i - \hat{y}_i)^2})}")
    st.write(
        "Reference link: https://openaccess.thecvf.com/content_CVPR_2019/papers/Barron_A_General_and_Adaptive_Robust_Loss_Function_CVPR_2019_paper.pdf")

    st.markdown("#### Partial derivatives")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1)}{\partial \beta_0} = (y_i - \hat{y}_i)^2  e^{-\frac{1}{2}(y_i - \hat{y}_i)^2}")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1)}{\partial \beta_1} = x_1  (y_i - \hat{y}_i)^2  e^{-\frac{1}{2}(y_i - \hat{y}_i)^2}")

    ############## NEW MODEL PREDICTION #####################

    st.subheader("New Model Prediction on Test Data")
    beta2 = reg2(X_train, y_train, verbose=verbosity)
    st.latex(fr"Price = {beta2[1]:.4f} \times MedInc + {beta2[0]:.4f}")

    pred_list = []
    for i in X_test:
        y_pred = i * beta2[1] + beta2[0]
        pred_list.append(y_pred)
    df_pred2 = pd.DataFrame(dict(y=y_test, y_pred=pred_list))
    st.write(df_pred2)

    err_list = []
    total_error = 0
    for i in range(df_pred2.shape[0]):
        err_ = 1 - math.exp((-(df_pred2['y'].iloc[i] - df_pred2['y_pred'].iloc[i]) ** 2) / 2)
        err_list.append(err_)
        total_error += err_
    temp2 = pd.DataFrame(dict(err=err_list, y_pred=df_pred2['y_pred']))

    fig = px.scatter(temp2, x="y_pred", y="err", trendline="ols", title="Loss Function vs. Prediction")
    st.plotly_chart(fig, use_container_width=True)
    mse2 = mean_squared_error(df_pred2.y, df_pred2.y_pred)
    mse2 = round(mse2, 3)
    st.write("Mean Squared Error of New Model's predictions = " + str(mse2))

    ### if we calculate total error according to the previous loss function:
    err_list2 = []
    total_error = 0
    for i in range(df_pred2.shape[0]):
        err = (df_pred2['y'].iloc[i] - df_pred2['y_pred'].iloc[i]) ** 2
        err_list2.append(err)
        total_error += err
    
    st.write("Total (mean squared) error = "+str(total_error))


    st.write("We can see that the new loss function is convex for beta values.")

    st.line_chart(df_pred2[-300:])


######################################################################

def reg(x, y, verbose=False):
    beta = np.random.random(2)

    if verbose:
        st.write(beta)
        st.write(x)

    alpha = 0.002
    my_bar = st.progress(0.)
    n_max_iter = 100
    for it in range(n_max_iter):

        err = 0
        for _x, _y in zip(x, y):
            y_pred = beta[0] + beta[1] * _x

            g_b0 = -2 * (_y - y_pred)
            g_b1 = -2 * ((_y - y_pred) * _x)

            # st.write(f"Gradient of beta0: {g_b0}")

            beta[0] = beta[0] - alpha * g_b0
            beta[1] = beta[1] - alpha * g_b1

            err += (_y - y_pred) ** 2

        print(f"{it} - Beta: {beta}, Error: {err}")
        my_bar.progress(it / n_max_iter)

    return beta


######################################################################

def reg2(x, y, verbose=False):
    beta = np.random.random(2)
    alpha = 0.0002
    n_max_iter = 1000

    for it in range(n_max_iter):
        err = 0
        for _x, _y in zip(x, y):
            y_pred = beta[0] + beta[1] * _x

            g_b0 = - (_y - y_pred) * math.exp((-(_y - y_pred) ** 2) / 2)
            g_b1 = - _x * (_y - y_pred) * math.exp((-(_y - y_pred) ** 2) / 2)

            beta[0] = beta[0] - alpha * g_b0
            beta[1] = beta[1] - alpha * g_b1

            unit_err = 1 - math.exp((-(_y - y_pred) ** 2) / 2)
            err += unit_err
    return beta


if __name__ == '__main__':
    main(st.sidebar.checkbox("verbosity"))