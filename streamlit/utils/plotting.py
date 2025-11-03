# utils/plotting.py
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_interactive_series(df_val, df_future, view_option="Completa"):
    df_plot = df_val.copy()

    # Agregar predicciones futuras
    df_future_ = df_future.copy()
    df_future_ = df_future_.rename(columns={'pred':'y_pred'})
    df_future_['y_real'] = np.nan
    df_plot = pd.concat([df_plot, df_future_], ignore_index=True)

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

    # Determinar ventana inicial según view_option
    last_day = df_plot['date'].max()
    if view_option == "Mensual":
        first_day = last_day - pd.Timedelta(days=30)
    elif view_option == "Semanal":
        first_day = last_day - pd.Timedelta(days=7)
    else:  # Completa
        first_day = df_plot['date'].min()

    # Configuración del eje x con rangeslider
    fig.update_xaxes(
        range=[first_day, last_day],      # ventana inicial
        rangeslider=dict(
            visible=True,
            thickness=0.02,
            bgcolor='#f0f0f0',
            borderwidth=0
        ),
        showline=True,
        mirror=True
    )

    # Limitar scroll vertical
    y_min = np.nanmin([df_plot['y_real'].min(), df_plot['y_pred'].min()])
    y_max = np.nanmax([df_plot['y_real'].max(), df_plot['y_pred'].max()])
    fig.update_yaxes(range=[y_min*0.95, y_max*1.05], fixedrange=True, showline=True, mirror=True)

    # Layout general
    fig.update_layout(
        template='plotly_white',
        title="Predicción vs Real",
        xaxis_title="Fecha",
        yaxis_title="Valor (€)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        margin=dict(l=20, r=20, t=30, b=40),
    )

    return fig
