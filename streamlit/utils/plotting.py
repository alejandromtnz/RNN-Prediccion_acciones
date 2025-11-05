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

    # Fecha a partir de la cual se resalta la predicción futura
    cutoff_date = pd.Timestamp("2025-11-01")

    # Datos antes y después del 1 de noviembre de 2025
    df_pred_hist = df_plot[df_plot['date'] < cutoff_date]
    df_pred_future = df_plot[df_plot['date'] >= cutoff_date]

    # Unir las dos series sin hueco (añadir último punto del tramo anterior al futuro)
    if not df_pred_hist.empty and not df_pred_future.empty:
        last_hist_point = df_pred_hist.iloc[[-1]]
        df_pred_future = pd.concat([last_hist_point, df_pred_future], ignore_index=True)

    # Decide si mostrar markers según el tamaño
    mode_real = 'lines' if len(df_plot) > 2000 else 'lines+markers'
    mode_pred = 'lines' if len(df_plot) > 2000 else 'lines+markers'

    # Serie real
    fig.add_trace(go.Scatter(
        x=df_plot['date'],
        y=df_plot['y_real'],
        mode=mode_real,
        name='Real',
        line=dict(color='#002b5c', width=2.5),
        marker=dict(size=4),
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Real:</b> %{y:.2f}€<extra></extra>'
    ))

    # Predicción hasta el 31 de octubre de 2025 (histórica)
    fig.add_trace(go.Scatter(
        x=df_pred_hist['date'],
        y=df_pred_hist['y_pred'],
        mode=mode_pred,
        name='Predicción (histórica)',
        line=dict(color='#b8860b', dash='dash', width=2.2),
        marker=dict(size=4),
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Predicción:</b> %{y:.2f}€<extra></extra>'
    ))

    # Predicción desde el 1 de noviembre de 2025 (futura)
    fig.add_trace(go.Scatter(
        x=df_pred_future['date'],
        y=df_pred_future['y_pred'],
        mode=mode_pred,
        name='Predicción futura',
        line=dict(color='#d4af37', dash='dash', width=2.8),  # dorado más brillante
        marker=dict(size=5),
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Predicción futura:</b> %{y:.2f}€<extra></extra>'
    ))


    # Ventana inicial según vista
    last_day = df_plot['date'].max()
    if view_option == "Último año":
        first_day = last_day - pd.Timedelta(days=365)
    elif view_option == "Último mes":
        first_day = last_day - pd.Timedelta(days=30)
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
            thickness=0.045,
            bgcolor='#f4f4f4',
            borderwidth=0,
        ),
        showline=True,
        mirror=True,
        linecolor="#ccc",
        tickfont=dict(family="Lato", size=12, color="#333"),
        title_font=dict(family="Cinzel", size=14, color="#111")
    )

    # Limitar desplazamiento vertical
    y_min = np.nanmin([df_plot['y_real'].min(), df_plot['y_pred'].min()])
    y_max = np.nanmax([df_plot['y_real'].max(), df_plot['y_pred'].max()])
    fig.update_yaxes(
        range=[y_min * 0.95, y_max * 1.05],
        fixedrange=True,
        showline=True,
        mirror=True,
        linecolor="#ccc",
        tickfont=dict(family="Lato", size=12, color="#333"),
        title_font=dict(family="Cinzel", size=14, color="#111")
    )

    # Layout general (estilo premium)
    fig.update_layout(
        template='plotly_white',
        title=dict(
            text="Predicción vs Real",
            font=dict(family="Cinzel", size=20, color="#111", weight="bold"),
            x=0.4,
            y=0.95
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.94,
            font=dict(family="Lato", size=13, color="#222")
        ),
        height=500,
        margin=dict(l=30, r=30, t=60, b=40),
        dragmode=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff"
    )

    return fig
