# utils/plotting.py
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_interactive_series(df_val, df_future, view_option="Completa"):
    df_plot = df_val.copy()

    # Agregar predicciones futuras
    df_future_ = df_future.copy()
    df_future_ = df_future_.rename(columns={'pred': 'y_pred'})
    df_future_['y_real'] = np.nan
    df_plot = pd.concat([df_plot, df_future_], ignore_index=True)

    # ⚡ Optimización: limitar puntos si hay demasiados
    max_points = 5000
    if len(df_plot) > max_points:
        step = len(df_plot) // max_points
        df_plot = df_plot.iloc[::step, :].reset_index(drop=True)

    # Crear figura
    fig = go.Figure()

    # Decide si mostrar markers según el tamaño
    mode_real = 'lines+markers' if len(df_plot) < 2000 else 'lines'
    mode_pred = 'lines+markers' if len(df_plot) < 2000 else 'lines'

    # Serie real
    fig.add_trace(go.Scatter(
        x=df_plot['date'],
        y=df_plot['y_real'],
        mode=mode_real,
        name='Real',
        line=dict(color='blue'),
        marker=dict(size=4),
        hovertemplate='Fecha: %{x}<br>Real: %{y:.2f}€<extra></extra>'
    ))

    # Serie de predicción
    fig.add_trace(go.Scatter(
        x=df_plot['date'],
        y=df_plot['y_pred'],
        mode=mode_pred,
        name='Predicción',
        line=dict(color='red', dash='dash'),
        marker=dict(size=4),
        hovertemplate='Fecha: %{x}<br>Predicción: %{y:.2f}€<extra></extra>'
    ))

    # Ventana inicial según vista
    last_day = df_plot['date'].max()
    if view_option == "Mensual":
        first_day = last_day - pd.Timedelta(days=30)
    elif view_option == "Semanal":
        first_day = last_day - pd.Timedelta(days=7)
    else:
        first_day = df_plot['date'].min()

    # Rango total fijo (lo que podrá moverse)
    total_min = df_plot['date'].min()
    total_max = df_plot['date'].max()

    # Configuración del eje X — mantiene barra, solo desplazamiento
    fig.update_xaxes(
        range=[first_day, last_day],
        fixedrange=True,
        rangeslider=dict(
            visible=True,
            thickness=0.04,
            bgcolor='#f0f0f0',
            borderwidth=0,
        ),
        showline=True,
        mirror=True
    )

    # Bloqueo visual del rango del slider (solo desplazamiento)
    fig.update_layout(
        xaxis_rangeslider_thickness=0.05,
        xaxis_rangeslider_bgcolor="#f4f4f4",
        xaxis_rangeslider_borderwidth=0,
        xaxis_rangeslider_range=[total_min, total_max],  # rango fijo del slider
        uirevision=True  # ⚡ evita que se redibuje entero al actualizar
    )

    # Limitar desplazamiento vertical
    y_min = np.nanmin([df_plot['y_real'].min(), df_plot['y_pred'].min()])
    y_max = np.nanmax([df_plot['y_real'].max(), df_plot['y_pred'].max()])
    fig.update_yaxes(
        range=[y_min * 0.95, y_max * 1.05],
        fixedrange=True,
        showline=True,
        mirror=True
    )

    # Layout general
    fig.update_layout(
        template='plotly_white',
        title="Predicción vs Real",
        xaxis_title="Fecha",
        yaxis_title="Valor (€)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        margin=dict(l=20, r=20, t=30, b=40),
        dragmode=False
    )

    return fig
