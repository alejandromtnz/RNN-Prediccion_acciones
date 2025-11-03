# utils/plotting.py
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_interactive_series(df_val, df_future, view_option="Completa"):
    """
    Devuelve gráfico interactivo con Plotly.
    df_val: DataFrame con columnas ['date','y_real','y_pred']
    df_future: DataFrame con columnas ['date','pred']
    view_option: 'Completa', 'Mensual', 'Semanal'
    """
    df_plot = df_val.copy()

    # Agregar predicciones futuras
    df_future_ = df_future.copy()
    df_future_ = df_future_.rename(columns={'pred':'y_pred'})
    df_future_['y_real'] = np.nan
    df_plot = pd.concat([df_plot, df_future_], ignore_index=True)

    # Filtrar según opción de vista
    if view_option == "Mensual":
        df_plot['month'] = df_plot['date'].dt.to_period('M')
        df_plot = df_plot.groupby('month').mean(numeric_only=True).reset_index()
        df_plot['date'] = df_plot['month'].dt.to_timestamp()
    elif view_option == "Semanal":
        df_plot['week'] = df_plot['date'].dt.to_period('W')
        df_plot = df_plot.groupby('week').mean(numeric_only=True).reset_index()
        df_plot['date'] = df_plot['week'].dt.start_time

    # Crear figura
    fig = go.Figure()

    # Real
    fig.add_trace(go.Scatter(
        x=df_plot['date'],
        y=df_plot['y_real'],
        mode='lines+markers',
        name='Real',
        line=dict(color='blue'),
        marker=dict(size=4),
        hovertemplate='Fecha: %{x}<br>Real: %{y:.2f}€<extra></extra>'
    ))

    # Predicción
    fig.add_trace(go.Scatter(
        x=df_plot['date'],
        y=df_plot['y_pred'],
        mode='lines+markers',
        name='Predicción',
        line=dict(color='red', dash='dash'),
        marker=dict(size=4),
        hovertemplate='Fecha: %{x}<br>Predicción: %{y:.2f}€<extra></extra>'
    ))

    fig.update_layout(
        template='plotly_white',
        title="Predicción vs Real",
        xaxis_title="Fecha",
        yaxis_title="Valor (€)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig
