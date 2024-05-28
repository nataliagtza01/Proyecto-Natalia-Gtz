import streamlit as st
from streamlit_option_menu import option_menu
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from requests.exceptions import ConnectionError
import time

# Función para cargar los datos desde la URL
def load_data(url):
    df = pd.read_csv(url)
    df['time'] = pd.to_datetime(df['time'])
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df

# Función para obtener los datos del VIX
def get_vix_data(period="1y"):
    vix = yf.Ticker("^VIX")
    vix_data = vix.history(period=period)
    vix_data.reset_index(inplace=True)
    vix_data['Date'] = pd.to_datetime(vix_data['Date'])  # Asegurarse de que 'Date' sea de tipo datetime
    vix_data['Date'] = vix_data['Date'].dt.strftime('%Y-%m-%d')
    return vix_data

# Función para ejecutar la regresión log-log
def perform_log_log_regression(X, Y):
    valid_idx = np.isfinite(X) & np.isfinite(Y)
    X = X[valid_idx]
    Y = Y[valid_idx]
    if len(X) == 0 or len(Y) == 0:
        raise ValueError("X and Y vectors must not be empty")
    X_log = np.log(X)
    Y_log = np.log(Y)
    coefficients = np.polyfit(X_log, Y_log, 1)
    m_log, b_log = coefficients
    Y_log_pred = m_log * X_log + b_log
    r2_log = r2_score(Y_log, Y_log_pred)
    return m_log, b_log, X_log, Y_log, Y_log_pred, r2_log

# Función para obtener datos con reintentos
def fetch_data_with_retry(fetch_function, max_retries=3, delay=2):
    retries = 0
    while retries < max_retries:
        try:
            return fetch_function()
        except (ConnectionError, TimeoutError) as e:
            retries += 1
            time.sleep(delay)
            if retries == max_retries:
                raise e

# Crear una barra de navegación personalizada
selected = option_menu(
    menu_title=None,  # No necesitamos un título en el menú
    options=["Introduction", "Stock Analysis", "Volatility Analysis"],  # Opciones del menú
    icons=["house", "bar-chart", "activity"],  # Iconos para las opciones
    menu_icon="cast",  # Icono del menú
    default_index=0,  # Opción seleccionada por defecto
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "navy", "font-size": "18px"}, 
        "nav-link": {"font-size": "18px", "text-align": "center", "margin": "0px", "color": "navy", "--hover-color": "#87CEEB"},
        "nav-link-selected": {"background-color": "#0000FF", "color": "white"},
    }
)

# Mostrar el contenido basado en la selección
if selected == "Introduction":
    st.title('Introduction')

    st.write('''
    <div style="text-align:center">
    <h1 style="color: #87CEEB; font-family: 'Arial Black', sans-serif; font-size: 36px;">USING SIMPLE REGRESSIONS FOR FINANCIAL ANALYSIS</h1>
    </div>
    ''', unsafe_allow_html=True)

    st.write('''
    ###### Natalia Gutiérrez Ahumada
    ###### 0233252
    ''')

    st.write('''
    # What is a regression analysis? :chart_with_upwards_trend:
    ### Regression is a statistical method used in finance, investing, and other disciplines that attempts to determine the relationship between one dependent variable and other explanatory variables (independent variables).
    ''')

    st.write('''
    # Simple regression
    ### *Linear regression* is the most common form of this technique, which establishes the linear relationship between two variables through a straight line to connect the best fitting line and the data point.
    ### Method: ordinary least squares (OLS)
    <div style="text-align: center; font-size: 32px; text-decoration: underline;">y = α + βx + error </div>
    ''', unsafe_allow_html=True)

    st.write('''
    ## Key Statistical Variables:
    - **Alpha (Intercept):** The expected mean value of Y when all X=0.
    - **Beta (Slope):** The change in Y for a one-unit change in X.
    - **R-squared:** A statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable.
    - **P-value:** The probability that the observed results would occur by random chance.
    - **Standard Error:** Measures the accuracy with which a sample distribution represents a population by using standard deviation.
    ''')

    st.write('''
    ## Explanation:
    - **Alpha (Intercept):** Indicates where the regression line intercepts the Y-axis. This value provides an estimation of the dependent variable when the independent variable is zero.
    - **Beta (Slope):** Represents the degree of change in the dependent variable for every one-unit change in the independent variable.
    - **R-squared:** Reflects the goodness of fit of the regression model. An R-squared value closer to 1 indicates a better fit.
    - **P-value:** Helps determine the significance of your results. A p-value less than 0.05 is commonly considered statistically significant.
    - **Standard Error:** A lower standard error indicates more precise estimates of the regression coefficients.
    ''')

    data = pd.read_excel("Housing.xlsx")
    columnas_disponibles = [col for col in data.columns if col != "House Price" and col != "State"]
    columna_seleccionada = st.selectbox("Selecciona una columna:", options=columnas_disponibles)
    precio_casa = data['House Price']
    columna_seleccionada_data = data[columna_seleccionada]
    data = pd.concat([precio_casa, columna_seleccionada_data], axis=1)

    X = data[columna_seleccionada]
    Y = data['House Price']

    fig, ax = plt.subplots()
    plt.scatter(X, Y)
    plt.xlabel(columna_seleccionada)
    plt.ylabel('House Price')
    plt.title('Gráfico de dispersión')
    st.pyplot(fig)

    X1 = sm.add_constant(X)
    reg = sm.OLS(Y, X1).fit()

    # Mostrar solo los datos estadísticos necesarios
    alpha, beta = reg.params
    r_squared = reg.rsquared
    p_value = reg.pvalues[1]
    std_err = reg.bse[1]

    st.write(f"Alpha (Intercept): {alpha:.4f}")
    st.write(f"Beta (Slope): {beta:.4f}")
    st.write(f"R-squared: {r_squared:.4f}")
    st.write(f"P-value: {p_value:.4f}")
    st.write(f"Standard Error: {std_err:.4f}")

elif selected == "Stock Analysis":
    st.title('Stock Analysis')

    # Cuadro de texto para ingresar el ticker de la acción
    ticker = st.text_input("Enter the stock ticker for analysis:")

    if ticker:
        try:
            stock_info = fetch_data_with_retry(lambda: yf.Ticker(ticker).info)
            if 'shortName' in stock_info:
                st.markdown(f"<h1 style='text-align: center;'>{stock_info['shortName']}</h1>", unsafe_allow_html=True)
            else:
                st.write("Short name not available.")
            st.markdown(f"<div style='font-size: smaller; margin-bottom: 20px;'>Description: {stock_info.get('longBusinessSummary', 'No description available.')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: smaller; margin-bottom: 20px;'>Sector: {stock_info.get('sector', 'N/A')}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: smaller; margin-bottom: 20px;'>Beta: {stock_info.get('beta', 'N/A')}</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            date_range = st.selectbox("Select date range:", ['5d', '3mo', '6mo', 'ytd', '1y', '5y'])

            stock_prices = fetch_data_with_retry(lambda: yf.download(ticker, period=date_range)['Close'])
            stock_prices = stock_prices / stock_prices.iloc[0] * 100  # Normalizar los datos

            if not stock_prices.empty:
                returns = stock_prices.pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Anualizar la volatilidad

                st.markdown(f"Standard Deviation (Daily): {returns.std():.4f}")
                st.markdown(f"Volatility (Annualized): {volatility:.4f}")

                # Mostrar rendimiento y riesgo
                performance = (stock_prices.iloc[-1] / stock_prices.iloc[0]) - 1
                st.markdown(f"Performance: {performance:.2%}")
                st.markdown(f"Risk (Volatility): {volatility:.2%}")

                fig, ax = plt.subplots()
                stock_prices.plot(ax=ax)
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.set_title(f'{ticker} Price Chart')
                st.pyplot(fig)

                indices = {
                    'S&P (GSPC)': '^GSPC',
                    'IPC (MXX)': '^MXX',
                    'Semiconductor (SOXX)': 'SOXX',
                    'Technology (IYW)': 'IYW',
                    'Software (IGV)': 'IGV',
                    'Financials (XLF)': 'XLF',
                    'Consumer Services (IYC)': 'IYC',
                    'Industrials (IYJ)': 'IYJ',
                    'Energy (XLE)': 'XLE',
                    'Health (XLV)': 'XLV',
                    'Materials (XLB)': 'XLB',
                    'Real Estate (XLRE)': 'XLRE',
                    'Telecom (IYZ)': 'IYZ'
                }

                selected_index = st.selectbox("Select an index:", list(indices.keys()))

                index_ticker = indices[selected_index]
                index_data = fetch_data_with_retry(lambda: yf.download(index_ticker, period=date_range)['Close'])
                index_data = index_data / index_data.iloc[0] * 100  # Normalizar los datos

                if not index_data.empty:
                    fig, ax = plt.subplots()
                    stock_prices.plot(ax=ax, label=ticker)
                    index_data.plot(ax=ax, label=selected_index)
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price')
                    ax.set_title(f'{ticker} vs {selected_index}')
                    ax.legend()
                    st.pyplot(fig)

                    # Comparativa con índices
                    index_returns = index_data.pct_change().dropna()
                    stock_returns = stock_prices.pct_change().dropna()

                    combined_data = pd.DataFrame({'Stock': stock_returns, 'Index': index_returns}).dropna()

                    if not combined_data.empty:
                        X = combined_data['Index']
                        Y = combined_data['Stock']

                        fig, ax = plt.subplots()
                        plt.scatter(X, Y)
                        plt.xlabel('Index Returns')
                        plt.ylabel('Stock Returns')
                        plt.title(f'Scatter plot of {ticker} vs {selected_index}')

                        # Trazar la línea de regresión
                        X1 = sm.add_constant(X)
                        reg = sm.OLS(Y, X1).fit()
                        ax.plot(X, reg.predict(X1), color='red', label='Regression Line')
                        ax.legend()
                        st.pyplot(fig)

                        # Mostrar solo los principales datos estadísticos
                        alpha, beta = reg.params
                        r_squared = reg.rsquared
                        std_err = reg.bse[1]

                        st.markdown(f"Alpha (Intercept): {alpha:.4f}")
                        st.markdown(f"Beta (Slope): {beta:.4f}")
                        st.markdown(f"R-squared: {r_squared:.4f}")
                        st.markdown(f"Standard Error: {std_err:.4f}")

                        # Mostrar la ecuación de la regresión
                        st.markdown(f"Regression Equation: y = {alpha:.4f} + {beta:.4f}x")

                        # Interpretación de la regresión
                        interpretation = f"""
                        **Interpretación:**
                        - **Beta (Pendiente):** {beta:.4f}. Esto indica que por cada 1% de cambio en el índice, se espera que el retorno del activo cambie en {beta:.4f}%.
                        - **Alpha (Intersección):** {alpha:.4f}. Esto sugiere que cuando el retorno del índice es cero, se espera que el retorno del activo sea {alpha:.4f}%.
                        - **R-squared:** {r_squared:.4f}. Este valor indica que el {r_squared * 100:.2f}% de la variabilidad en el retorno del activo puede ser explicada por el retorno del índice.
                        - **Error estándar:** {std_err:.4f}. Esto mide la precisión de la estimación de la pendiente. Un error estándar más bajo indica una estimación más precisa.
                        """
                        st.markdown(interpretation)

                        # Botón para encontrar el índice con mayor correlación y significancia
                        if st.button("Find Most Correlated Index"):
                            max_corr = -1
                            best_index = None
                            
                            for idx_name, idx_ticker in indices.items():
                                index_data = fetch_data_with_retry(lambda: yf.download(idx_ticker, period=date_range)['Close'])
                                index_data = index_data / index_data.iloc[0] * 100
                                
                                if not index_data.empty:
                                    index_returns = index_data.pct_change().dropna()
                                    stock_returns = stock_prices.pct_change().dropna()

                                    combined_data = pd.DataFrame({'Stock': stock_returns, 'Index': index_returns}).dropna()

                                    if not combined_data.empty:
                                        X = combined_data['Index']
                                        Y = combined_data['Stock']

                                        X1 = sm.add_constant(X)
                                        reg = sm.OLS(Y, X1).fit()
                                        corr = reg.rsquared

                                        if corr > max_corr:
                                            max_corr = corr
                                            best_index = idx_name

                            if best_index:
                                st.write(f"The index with the highest correlation is: {best_index}")
                                st.write(f"Correlation: {max_corr:.4f}")
                            else:
                                st.write("No significant correlation found.")

                        # Gráfica de la distribución de los errores
                        residuals = reg.resid
                        fig, ax = plt.subplots()
                        sns.histplot(residuals, kde=True, ax=ax)
                        ax.set_title('Distribución de los Errores')
                        ax.set_xlabel('Errores')
                        ax.set_ylabel('Frecuencia')
                        st.pyplot(fig)
                    else:
                        st.write("Not enough data to perform regression analysis.")
                else:
                    st.write("Index data is empty.")
            else:
                st.write("Stock data is empty.")
        except Exception as e:
            st.error(f"Error al obtener los datos: {str(e)}. Please try again later.")

elif selected == "Volatility Analysis":
    st.header('Implied Volatility Analysis')
    
    st.write('''
    ## Implied Volatility
    Implied volatility is the market's forecast of a likely movement in a security's price. It is a metric used by investors to estimate future fluctuations (volatility) of a security's price based on certain predictive factors. Implied volatility is denoted by the symbol σ (sigma). It can often be thought to be a proxy of market risk.

    ## Implied Volatility and Options
    Implied volatility is one of the deciding factors in the pricing of options. Buying options contracts allow the holder to buy or sell an asset at a specific price during a pre-determined period. Implied volatility approximates the future value of the option, and the option's current value is also taken into consideration. Options with high implied volatility have higher premiums and vice versa.
    ''')

    # Cargar los datos desde la URL
    url = "https://research-watchlists.s3.amazonaws.com/df_UniversidadPanamericana_ohlc.csv"
    df = load_data(url)

    # Filtrar los datos del último año
    one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)
    df_last_year = df[df['time'] >= one_year_ago]

    # Seleccionar símbolos
    symbols = df_last_year['Symbol'].unique()
    selected_symbol = st.selectbox("Select Symbol:", symbols)

    # Obtener datos históricos del VIX
    vix_data = get_vix_data()

    # Inicializar la figura de Plotly
    fig = go.Figure()

    # Graficar la línea de tendencia para el VIX
    fig.add_trace(go.Scatter(
        x=vix_data['Date'],
        y=vix_data['Close'],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name='VIX'
    ))

    if selected_symbol:
        symbol_data = df_last_year[df_last_year['Symbol'] == selected_symbol]
        symbol_data.sort_values(by='time', ascending=True, inplace=True)
        symbol_data['time'] = symbol_data['time'].dt.strftime('%Y-%m-%d')
        symbol_data['annualized_volatility'] = symbol_data['impVolatility'] * np.sqrt(252)
        
        # Añadir traza a la figura de Plotly
        fig.add_trace(go.Scatter(
            x=symbol_data['time'],
            y=symbol_data['annualized_volatility'],
            mode='lines+markers',
            name=f"{selected_symbol} Annualized Volatility"
        ))

        all_volatilities = symbol_data['annualized_volatility'].values
        all_vix_values = vix_data['Close'].values[:len(symbol_data)]  # Asegurarse de que las longitudes coincidan

        # Mostrar la gráfica interactiva en Streamlit
        st.plotly_chart(fig)

        # Ejecución de la regresión log-log
        st.subheader("Ejecución de la Regresión Log-Log")

        # Convertir listas a arrays de numpy y eliminar NaNs/infs
        X = np.array(all_volatilities)
        Y = np.array(all_vix_values)

        # Filtrar NaNs e infinitos
        valid_idx = np.isfinite(X) & np.isfinite(Y)
        X = X[valid_idx]
        Y = Y[valid_idx]

        # Aplicar la transformación logarítmica
        X_log = np.log(X)
        Y_log = np.log(Y)

        try:
            # Calcular los coeficientes de la regresión lineal en los datos logarítmicos
            coefficients = np.polyfit(X_log, Y_log, 1)
            m_log, b_log = coefficients

            # Predecir los valores de Y utilizando el modelo
            Y_log_pred = m_log * X_log + b_log

            # Calcular el R²
            r2_log = r2_score(Y_log, Y_log_pred)

            # Crear la gráfica de dispersión con la línea de regresión lineal
            fig, ax = plt.subplots()
            ax.scatter(X_log, Y_log, label='Datos')
            ax.plot(X_log, Y_log_pred, color='red', label=f'Regresión log-log: y = {m_log:.2f}x + {b_log:.2f}')
            
            # Configurar los ejes y etiquetas
            ax.set_ylabel('Log(VIX)')
            ax.set_xlabel('Log(Annualized Volatility)')
            ax.set_title('VIX vs Annualized Volatility (Log-Log)')
            ax.legend()
            
            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)

            # Transformar los coeficientes a la escala original
            alpha_original = np.exp(b_log)
            beta_original = m_log

            # Mostrar los coeficientes en Streamlit
            st.write(f"Coeficiente de la pendiente (beta): {beta_original:.2f}")
            st.write(f"Intersección con el eje Y (alpha, en la escala original): {alpha_original:.2f}")
            st.write(f"Coeficiente de determinación (R²): {r2_log:.2f}")
            
            # Calcular los residuos
            residuals_log = Y_log - Y_log_pred

            # Crear una figura para la distribución de los residuos
            fig, ax = plt.subplots()
            sns.histplot(residuals_log, kde=True, ax=ax)
            ax.set_title('Distribución de Probabilidad de los Residuos (Log-Log)')
            ax.set_xlabel('Residuos')
            ax.set_ylabel('Densidad')

            # Mostrar la gráfica de distribución de residuos en Streamlit
            st.pyplot(fig)
            
            # Mostrar estadísticas de los residuos
            st.write(f"Media de los residuos: {np.mean(residuals_log):.2f}")
            st.write(f"Desviación estándar de los residuos: {np.std(residuals_log):.2f}")

        except Exception as e:
            st.error(f"Error en la ejecución de la regresión log-log: {e}")

        # Ejecución de la regresión de cambios porcentuales
        st.subheader("Ejecución de la Regresión de Cambios Porcentuales")

        try:
            # Calcular los retornos porcentuales
            pct_change_volatility = np.diff(all_volatilities) / all_volatilities[:-1] * 100
            pct_change_vix = np.diff(all_vix_values) / all_vix_values[:-1] * 100

            # Filtrar NaNs e infinitos
            valid_idx = np.isfinite(pct_change_volatility) & np.isfinite(pct_change_vix)
            X_pct = pct_change_volatility[valid_idx]
            Y_pct = pct_change_vix[valid_idx]

            if len(X_pct) == 0 or len(Y_pct) == 0:
                raise ValueError("Insufficient data for percentage change regression")

            # Calcular los coeficientes de la regresión lineal en los cambios porcentuales
            X1_pct = sm.add_constant(X_pct)
            reg_pct = sm.OLS(Y_pct, X1_pct).fit()

            # Mostrar los coeficientes y estadísticas de la regresión
            alpha_pct, beta_pct = reg_pct.params
            r_squared_pct = reg_pct.rsquared
            residuals_pct = reg_pct.resid

            st.write(f"Coeficiente de la pendiente (beta): {beta_pct:.2f}")
            st.write(f"Intersección con el eje Y (alpha): {alpha_pct:.2f}")
            st.write(f"Coeficiente de determinación (R²): {r_squared_pct:.2f}")

            # Crear la gráfica de dispersión con la línea de regresión
            fig, ax = plt.subplots()
            ax.scatter(X_pct, Y_pct, label='Datos')
            ax.plot(X_pct, reg_pct.predict(X1_pct), color='red', label=f'Regresión cambios porcentuales: y = {alpha_pct:.2f} + {beta_pct:.2f}x')
            
            # Configurar los ejes y etiquetas
            ax.set_ylabel('VIX % Change')
            ax.set_xlabel('Volatility % Change')
            ax.set_title('VIX vs Volatility (Percentage Changes)')
            ax.legend()
            
            # Mostrar la gráfica en Streamlit
            st.pyplot(fig)

            # Crear una figura para la distribución de los residuos
            fig, ax = plt.subplots()
            sns.histplot(residuals_pct, kde=True, ax=ax)
            ax.set_title('Distribución de Probabilidad de los Residuos (Cambios Porcentuales)')
            ax.set_xlabel('Residuos')
            ax.set_ylabel('Densidad')

            # Mostrar la gráfica de distribución de residuos en Streamlit
            st.pyplot(fig)

            # Mostrar estadísticas de los residuos
            st.write(f"Media de los residuos: {np.mean(residuals_pct):.2f}")
            st.write(f"Desviación estándar de los residuos: {np.std(residuals_pct):.2f}")

        except Exception as e:
            st.error(f"Error en la ejecución de la regresión de cambios porcentuales: {e}")

    st.write("----")
