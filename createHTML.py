import pandas as pd
import numpy as np

from bokeh.io import output_file, save
from bokeh.plotting import figure
from bokeh.layouts import column, row

from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Span,
    NumeralTickFormatter,
    DataTable,
    TableColumn,
    NumberFormatter,
)
from bokeh.models.widgets import StringFormatter
from bokeh.transform import dodge
from bokeh.models.widgets import HTMLTemplateFormatter
from bokeh.models.widgets import Div
from bokeh.models import FactorRange


# ============================================================
#           GENERACIÓN DE TRADES FAKE (TAL CUAL)
# ============================================================

def generate_fake_trades(n_trades: int = 100000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Generar ISINs ficticios (formato tipo alemán: DE + 10 caracteres alfanuméricos)
    def random_isin(rng, prefix="DE"):
        chars = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        return prefix + "".join(rng.choice(chars, size=10))

    # Pool de ISINs
    isins = [random_isin(rng) for _ in range(20)]

    # Rango temporal
    start_ts = pd.Timestamp("2024-01-01 09:00:00").value
    end_ts = pd.Timestamp("2025-11-30 22:00:00").value

    trade_times = pd.to_datetime(
        rng.integers(start_ts, end_ts, size=n_trades)
    )

    # quantity: float con signo
    magnitude = rng.uniform(1.0, 100001.0, size=n_trades)  # siempre > 0
    sign = rng.choice([-1.0, 1.0], size=n_trades)
    quantity = magnitude * sign  # float con signo

    # Precio "oculto" para generar un vol coherente
    price = rng.uniform(1.0, 300.0, size=n_trades)  # floats
    vol = price * quantity  # también float con signo

    # buyOrSell en función del signo de quantity
    buy_or_sell = np.where(quantity > 0, "Buy", "Sell")

    df = pd.DataFrame({
        "isin": rng.choice(isins, size=n_trades),
        "tradeTime": trade_times,
        "buyOrSell": buy_or_sell,
        "quantity": quantity,
        "vol": vol,
    })

    return df


df = generate_fake_trades()

df = df.copy()
df["tradeTime"] = pd.to_datetime(df["tradeTime"])
df["month"] = df["tradeTime"].dt.to_period("M").dt.to_timestamp()
df["date"] = df["tradeTime"].dt.normalize()
df["abs_vol"] = df["vol"].abs()
df["trade_sign"] = np.where(df["buyOrSell"] == "Buy", 1, -1)


# ============================================================
#   MONTHLY MASTER (VOL + TRADES + PERCENTILES) TAL CUAL
# ============================================================

months = np.sort(df["month"].unique())
monthly = pd.DataFrame({"month": months})

# Volumen bruto por lado
vol_side = (
    df.groupby(["month", "buyOrSell"])["abs_vol"]
      .sum()
      .unstack("buyOrSell")
      .fillna(0.0)
      .reset_index()
)
vol_side["gross_buy"] = vol_side.get("Buy", 0.0)
vol_side["gross_sell"] = vol_side.get("Sell", 0.0)
vol_side = vol_side[["month", "gross_buy", "gross_sell"]]

# Volumen neto
net_vol = df.groupby("month")["vol"].sum().reset_index(name="net_volume")

# Trades por lado
trades_side = (
    df.groupby(["month", "buyOrSell"])["isin"]
      .size()
      .unstack("buyOrSell")
      .fillna(0)
      .reset_index()
)
trades_side["buy_trades"] = trades_side.get("Buy", 0)
trades_side["sell_trades"] = trades_side.get("Sell", 0)
trades_side["total_trades"] = trades_side["buy_trades"] + trades_side["sell_trades"]
trades_side = trades_side[["month", "buy_trades", "sell_trades", "total_trades"]]

# Net trades
net_trades = (
    df.groupby("month")["trade_sign"]
      .sum()
      .reset_index(name="net_trades")
)

# Percentiles volumen |vol| por trade
vol_pct = (
    df.groupby("month")["abs_vol"]
      .agg(
          v_p10=lambda s: np.percentile(s, 10),
          v_p25=lambda s: np.percentile(s, 25),
          v_p50=lambda s: np.percentile(s, 50),
          v_p75=lambda s: np.percentile(s, 75),
          v_p90=lambda s: np.percentile(s, 90),
          v_mean="mean",
      )
      .reset_index()
)

# Percentiles nº trades diarios por mes
daily_counts = (
    df.groupby(["month", "date"])["isin"]
      .size()
      .reset_index(name="n_trades")
)
trades_pct = (
    daily_counts.groupby("month")["n_trades"]
      .agg(
          t_p10=lambda s: np.percentile(s, 10),
          t_p25=lambda s: np.percentile(s, 25),
          t_p50=lambda s: np.percentile(s, 50),
          t_p75=lambda s: np.percentile(s, 75),
          t_p90=lambda s: np.percentile(s, 90),
          t_mean="mean",
      )
      .reset_index()
)

monthly = (
    monthly
    .merge(vol_side, on="month", how="left")
    .merge(net_vol, on="month", how="left")
    .merge(trades_side, on="month", how="left")
    .merge(net_trades, on="month", how="left")
    .merge(vol_pct, on="month", how="left")
    .merge(trades_pct, on="month", how="left")
)

for c in monthly.columns:
    if c != "month":
        monthly[c] = monthly[c].fillna(0)

monthly["gross_total"] = monthly["gross_buy"] + monthly["gross_sell"]
monthly["month_label"] = monthly["month"].dt.strftime("%Y-%b")

def fmt_num(x):
    return f"{x:,.2f}"

# Strings numéricos
monthly["gross_buy_str"] = monthly["gross_buy"].apply(fmt_num)
monthly["gross_sell_str"] = monthly["gross_sell"].apply(fmt_num)
monthly["gross_total_str"] = monthly["gross_total"].apply(fmt_num)
monthly["net_volume_str"] = monthly["net_volume"].apply(fmt_num)

monthly["buy_trades_str"] = monthly["buy_trades"].astype(float).apply(fmt_num)
monthly["sell_trades_str"] = monthly["sell_trades"].astype(float).apply(fmt_num)
monthly["total_trades_str"] = monthly["total_trades"].astype(float).apply(fmt_num)

monthly["v_p10_str"] = monthly["v_p10"].apply(fmt_num)
monthly["v_p25_str"] = monthly["v_p25"].apply(fmt_num)
monthly["v_p50_str"] = monthly["v_p50"].apply(fmt_num)
monthly["v_p75_str"] = monthly["v_p75"].apply(fmt_num)
monthly["v_p90_str"] = monthly["v_p90"].apply(fmt_num)
monthly["v_mean_str"] = monthly["v_mean"].apply(fmt_num)

monthly["t_p10_str"] = monthly["t_p10"].apply(fmt_num)
monthly["t_p25_str"] = monthly["t_p25"].apply(fmt_num)
monthly["t_p50_str"] = monthly["t_p50"].apply(fmt_num)
monthly["t_p75_str"] = monthly["t_p75"].apply(fmt_num)
monthly["t_p90_str"] = monthly["t_p90"].apply(fmt_num)
monthly["t_mean_str"] = monthly["t_mean"].apply(fmt_num)

monthly["net_color"] = np.where(monthly["net_volume"] >= 0, "green", "red")
monthly["net_volume_str"] = monthly["net_volume"].apply(fmt_num)

monthly["net_volume_is_max"] = monthly["net_volume"] == monthly["net_volume"].max()
monthly["net_volume_is_min"] = monthly["net_volume"] == monthly["net_volume"].min()

monthly["year_str"] = monthly["month"].dt.strftime("%Y")
monthly["mon_str"] = monthly["month"].dt.strftime("%b")
monthly["month_label"] = monthly["month"].dt.strftime("%Y-%b")
monthly["month_factor"] = list(zip(monthly["year_str"], monthly["mon_str"]))

# Flags max/min para varios campos
for col in ["gross_buy", "gross_sell", "gross_total", "net_volume"]:
    monthly[f"{col}_is_max"] = monthly[col] == monthly[col].max()
    monthly[f"{col}_is_min"] = monthly[col] == monthly[col].min()

for col in ["buy_trades", "sell_trades", "total_trades"]:
    monthly[f"{col}_is_max"] = monthly[col] == monthly[col].max()
    monthly[f"{col}_is_min"] = monthly[col] == monthly[col].min()

for col in [
    "v_p10", "v_p25", "v_p50", "v_p75", "v_p90", "v_mean",
    "t_p10", "t_p25", "t_p50", "t_p75", "t_p90", "t_mean",
]:
    monthly[f"{col}_is_max"] = monthly[col] == monthly[col].max()
    monthly[f"{col}_is_min"] = monthly[col] == monthly[col].min()

factors = monthly["month_factor"].tolist()
month_xrange = FactorRange(*factors)

fmt_axis_vol = NumeralTickFormatter(format="0a")
fmt_axis_tr = NumeralTickFormatter(format="0a")  # ya no se usa, pero no molesta

source_month = ColumnDataSource(monthly)


# ============================================================
# ROW 1 – MONTHLY TRADING VOLUME (GROSS) + TABLA
# ============================================================

p_vol_gross = figure(
    x_range=month_xrange,
    width=650,
    height=350,
    title="Monthly Trading Volume – Buy, Sell and Total",
    toolbar_location="right",
    tools="pan,box_zoom,reset,save",
)

r_buy_vol = p_vol_gross.vbar(
    x=dodge("month_factor", -0.18, range=p_vol_gross.x_range),
    top="gross_buy",
    width=0.35,
    source=source_month,
    legend_label="Buy volume",
    fill_color="#1f77b4",
)

r_sell_vol = p_vol_gross.vbar(
    x=dodge("month_factor", +0.18, range=p_vol_gross.x_range),
    top="gross_sell",
    width=0.35,
    source=source_month,
    legend_label="Sell volume",
    fill_color="#ff7f0e",
)

p_vol_gross.line(
    "month_factor",
    "gross_total",
    source=source_month,
    line_width=2,
    legend_label="Total volume",
    line_color="black",
)

r_total_vol = p_vol_gross.scatter(
    "month_factor",
    "gross_total",
    source=source_month,
    size=6,
    fill_color="black",
    legend_label="Total volume",
)

p_vol_gross.xaxis.major_label_orientation = 0.9
p_vol_gross.yaxis.formatter = fmt_axis_vol

legend_vg = p_vol_gross.legend[0]
p_vol_gross.add_layout(legend_vg, "right")
p_vol_gross.legend.click_policy = "hide"

# --- Hover ÚNICO (sin duplicados) ---
hover_vol_gross = HoverTool(
    renderers=[r_buy_vol, r_sell_vol, r_total_vol],
    tooltips=[
        ("Month", "@month_label"),
        ("Buy volume", "@gross_buy{0,0.00}"),
        ("Sell volume", "@gross_sell{0,0.00}"),
        ("Total volume", "@gross_total{0,0.00}"),
    ],
)
p_vol_gross.add_tools(hover_vol_gross)

# --- Tabla gross volume + net volume (TODO en una sola tabla) ---

template_gross_buy = """
<% var zebra = (index % 2 === 0) ? '#ffffff' : '#f7f7f7';
   var bg = zebra;
   if (gross_buy_is_max) { bg = '#e0f2ff'; }
   else if (gross_buy_is_min) { bg = '#ffeaea'; } %>
<div style="background-color:<%= bg %>; text-align:right;">
  <%= gross_buy_str %>
</div>
"""

template_gross_sell = """
<% var zebra = (index % 2 === 0) ? '#ffffff' : '#f7f7f7';
   var bg = zebra;
   if (gross_sell_is_max) { bg = '#e0f2ff'; }
   else if (gross_sell_is_min) { bg = '#ffeaea'; } %>
<div style="background-color:<%= bg %>; text-align:right;">
  <%= gross_sell_str %>
</div>
"""

template_gross_total = """
<% var zebra = (index % 2 === 0) ? '#ffffff' : '#f7f7f7';
   var bg = zebra;
   if (gross_total_is_max) { bg = '#e0f2ff'; }
   else if (gross_total_is_min) { bg = '#ffeaea'; } %>
<div style="background-color:<%= bg %>; text-align:right;">
  <%= gross_total_str %>
</div>
"""

# NET volume con color verde/rojo según signo
template_net_vol_cell = """
<% var zebra = (index % 2 === 0) ? '#ffffff' : '#f7f7f7';
   var bg = zebra;
   if (net_volume_is_max) { bg = '#e0f2ff'; }
   else if (net_volume_is_min) { bg = '#ffeaea'; } %>
<div style="background-color:<%= bg %>; text-align:right; color:<%= net_color %>;">
  <%= net_volume_str %>
</div>
"""


columns_gross = [
    TableColumn(
        field="month_label",
        title="Month",
        formatter=StringFormatter(text_align="center"),
    ),
    TableColumn(
        field="gross_buy_str",
        title="Buy volume",
        formatter=HTMLTemplateFormatter(template=template_gross_buy),
    ),
    TableColumn(
        field="gross_sell_str",
        title="Sell volume",
        formatter=HTMLTemplateFormatter(template=template_gross_sell),
    ),
    TableColumn(
        field="gross_total_str",
        title="Total volume",
        formatter=HTMLTemplateFormatter(template=template_gross_total),
    ),
    TableColumn(  # <<< CAMBIO: nueva columna net volume
        field="net_volume_str",
        title="Net volume",
        formatter=HTMLTemplateFormatter(template=template_net_vol_cell),
    ),
]

table_gross = DataTable(
    source=source_month,
    columns=columns_gross,
    width=1300,   # <<< CAMBIO: ancho grande para cubrir ambos gráficos
    height=350,
    index_position=None,
)


# ============================================================
# MONTHLY NET VOLUME (SOLO GRÁFICO, SIN TABLA PROPIA)
# ============================================================

monthly["net_volume_is_max"] = monthly["net_volume"] == monthly["net_volume"].max()
monthly["net_volume_is_min"] = monthly["net_volume"] == monthly["net_volume"].min()
source_month.data = ColumnDataSource.from_df(monthly)

p_vol_net = figure(
    x_range=month_xrange,
    width=650,
    height=350,   # <<< CAMBIO: misma altura que p_vol_gross para alinearlos
    title="Monthly Net Trading Volume",
    toolbar_location="right",
    tools="pan,box_zoom,reset,save",   # sin 'hover'
)

# Línea horizontal en 0
p_vol_net.add_layout(
    Span(location=0, dimension="width", line_dash="dashed", line_width=1)
)

# HISTOGRAMA (barras) neto
r_net_bar = p_vol_net.vbar(
    x="month_factor",
    top="net_volume",
    width=0.6,
    source=source_month,
    fill_color="net_color",
    legend_label="Net volume",  # <<< CAMBIO: para tener leyenda
)

p_vol_net.yaxis.formatter = fmt_axis_vol
p_vol_net.xaxis.major_label_orientation = 0.9

hover_vol_net = HoverTool(
    renderers=[r_net_bar],
    tooltips=[
        ("Month", "@month_label"),
        ("Net volume", "@net_volume{0,0.00}"),
    ],
)
p_vol_net.add_tools(hover_vol_net)

# <<< CAMBIO: leyenda fuera del gráfico, alineada como en p_vol_gross
legend_vn = p_vol_net.legend[0]
p_vol_net.add_layout(legend_vn, "right")
p_vol_net.legend.click_policy = "hide"


# ============================================================
# VOLUME DISTRIBUTION BY TRADE SIZE (VOLÚMENES) + TABLA
# (Buckets definidos por abs(vol) de CADA trade)
# ============================================================

df_vol = df.copy()
df_vol["abs_vol"] = df_vol["vol"].abs()

# Buckets por ABS(VOL) del trade
bin_edges = [1, 100, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000, np.inf]
bin_labels = [
    "1–100",
    "101–500",
    "501–1,000",
    "1,001–2,000",
    "2,001–5,000",
    "5,001–10,000",
    "10,001–25,000",
    "25,001–50,000",
    "50,001–100,000",
    "100,001+",
]

df_vol["vol_bucket"] = pd.cut(
    df_vol["abs_vol"],
    bins=bin_edges,
    labels=bin_labels,
    right=False,
    include_lowest=True,
)

# ----------------- 1) DISTRIBUCIÓN DE VOLUMEN -----------------
vol_side_bucket = (
    df_vol.groupby(["vol_bucket", "buyOrSell"])["abs_vol"]
          .sum()
          .unstack("buyOrSell")
          .reindex(bin_labels)
          .fillna(0.0)
          .reset_index()
          .rename(columns={"vol_bucket": "bucket"})
)

vol_side_bucket["buy_volume"]  = vol_side_bucket.get("Buy", 0.0)
vol_side_bucket["sell_volume"] = vol_side_bucket.get("Sell", 0.0)
vol_side_bucket["total_volume"] = (
    vol_side_bucket["buy_volume"] + vol_side_bucket["sell_volume"]
)

total_volume_all = vol_side_bucket["total_volume"].sum()

vol_side_bucket["total_share"] = np.where(
    total_volume_all > 0,
    vol_side_bucket["total_volume"] * 100.0 / total_volume_all,
    0.0,
)
vol_side_bucket["buy_share"] = np.where(
    vol_side_bucket["total_volume"] > 0,
    vol_side_bucket["buy_volume"] * 100.0 / vol_side_bucket["total_volume"],
    0.0,
)
vol_side_bucket["sell_share"] = np.where(
    vol_side_bucket["total_volume"] > 0,
    vol_side_bucket["sell_volume"] * 100.0 / vol_side_bucket["total_volume"],
    0.0,
)

vol_side_bucket["sell_top"] = (
    vol_side_bucket["buy_volume"] + vol_side_bucket["sell_volume"]
)

# Strings para la tabla de VOLUMEN
vol_side_bucket["total_volume_str"] = vol_side_bucket["total_volume"].apply(fmt_num)
vol_side_bucket["total_share_str"]  = vol_side_bucket["total_share"].map(lambda x: f"{x:,.2f}%")
vol_side_bucket["buy_volume_str"]   = vol_side_bucket["buy_volume"].apply(fmt_num)
vol_side_bucket["sell_volume_str"]  = vol_side_bucket["sell_volume"].apply(fmt_num)
vol_side_bucket["buy_share_str"]    = vol_side_bucket["buy_share"].map(lambda x: f"{x:,.2f}%")
vol_side_bucket["sell_share_str"]   = vol_side_bucket["sell_share"].map(lambda x: f"{x:,.2f}%")

vol_side_bucket["total_vol_is_max"] = (
    vol_side_bucket["total_volume"] == vol_side_bucket["total_volume"].max()
)
vol_side_bucket["total_vol_is_min"] = (
    vol_side_bucket["total_volume"] == vol_side_bucket["total_volume"].min()
)
vol_side_bucket["share_is_max"] = (
    vol_side_bucket["total_share"] == vol_side_bucket["total_share"].max()
)
vol_side_bucket["share_is_min"] = (
    vol_side_bucket["total_share"] == vol_side_bucket["total_share"].min()
)

source_vdist = ColumnDataSource(vol_side_bucket)

fmt_axis_bucket = NumeralTickFormatter(format="0a")

p_vol_dist = figure(
    x_range=bin_labels,
    width=650,
    height=350,
    title="Volume Distribution by Trade Size (Buckets by |vol| per trade)",
    toolbar_location="right",
    tools="pan,box_zoom,reset,save",
)

r_buy_vd = p_vol_dist.vbar(
    x="bucket",
    top="buy_volume",
    width=0.8,
    source=source_vdist,
    fill_color="#1f77b4",
    legend_label="Buy volume",
)

r_sell_vd = p_vol_dist.vbar(
    x="bucket",
    bottom="buy_volume",
    top="sell_top",
    width=0.8,
    source=source_vdist,
    fill_color="#ff7f0e",
    legend_label="Sell volume",
)

p_vol_dist.yaxis.axis_label = "Volume (sum of |vol|)"
p_vol_dist.yaxis.formatter = fmt_axis_bucket
p_vol_dist.xaxis.major_label_orientation = 0.85

legend_vd = p_vol_dist.legend[0]
p_vol_dist.add_layout(legend_vd, "right")
p_vol_dist.legend.click_policy = "hide"

hover_vol_dist = HoverTool(
    renderers=[r_buy_vd, r_sell_vd],
    tooltips=[
        ("Bucket (by |vol|)", "@bucket"),
        ("Total volume", "@total_volume{0,0.00}"),
        ("Share of volume", "@total_share{0,0.00}%"),
        ("Buy volume", "@buy_volume{0,0.00}"),
        ("Sell volume", "@sell_volume{0,0.00}"),
        ("Buy share in bucket", "@buy_share{0,0.00}%"),
        ("Sell share in bucket", "@sell_share{0,0.00}%"),
    ],
)
p_vol_dist.add_tools(hover_vol_dist)

# --------- Tabla de VOLUMEN ----------
template_total_vol_b = """
<% var zebra = (index % 2 === 0) ? '#ffffff' : '#f7f7f7';
   var bg = zebra;
   if (total_vol_is_max) { bg = '#e0f2ff'; }
   else if (total_vol_is_min) { bg = '#ffeaea'; } %>
<div style="background-color:<%= bg %>; text-align:right;">
  <%= total_volume_str %>
</div>
"""

template_total_share_b = """
<% var zebra = (index % 2 === 0) ? '#ffffff' : '#f7f7f7';
   var bg = zebra;
   if (share_is_max) { bg = '#e0f2ff'; }
   else if (share_is_min) { bg = '#ffeaea'; } %>
<div style="background-color:<%= bg %>; text-align:right;">
  <%= total_share_str %>
</div>
"""

template_buy_vol_b = """
<% var zebra = (index % 2 === 0) ? '#ffffff' : '#f7f7f7'; %>
<div style="background-color:<%= zebra %>; text-align:right;">
  <%= buy_volume_str %>
</div>
"""

template_sell_vol_b = """
<% var zebra = (index % 2 === 0) ? '#ffffff' : '#f7f7f7'; %>
<div style="background-color:<%= zebra %>; text-align:right;">
  <%= sell_volume_str %>
</div>
"""

template_buy_share_b = """
<% var zebra = (index % 2 === 0) ? '#ffffff' : '#f7f7f7'; %>
<div style="background-color:<%= zebra %>; text-align:right;">
  <%= buy_share_str %>
</div>
"""

template_sell_share_b = """
<% var zebra = (index % 2 === 0) ? '#ffffff' : '#f7f7f7'; %>
<div style="background-color:<%= zebra %>; text-align:right;">
  <%= sell_share_str %>
</div>
"""

columns_vdist = [
    TableColumn(field="bucket",           title="Bucket (by |vol|)",
                formatter=StringFormatter(text_align="center")),
    TableColumn(field="total_volume_str", title="Total volume",
                formatter=HTMLTemplateFormatter(template=template_total_vol_b)),
    TableColumn(field="total_share_str",  title="Share of volume",
                formatter=HTMLTemplateFormatter(template=template_total_share_b)),
    TableColumn(field="buy_volume_str",   title="Buy volume",
                formatter=HTMLTemplateFormatter(template=template_buy_vol_b)),
    TableColumn(field="sell_volume_str",  title="Sell volume",
                formatter=HTMLTemplateFormatter(template=template_sell_vol_b)),
    TableColumn(field="buy_share_str",    title="Buy %",
                formatter=HTMLTemplateFormatter(template=template_buy_share_b)),
    TableColumn(field="sell_share_str",   title="Sell %",
                formatter=HTMLTemplateFormatter(template=template_sell_share_b)),
]

table_vdist = DataTable(
    source=source_vdist,
    columns=columns_vdist,
    width=450,
    height=350,
    index_position=None,
)

# ============================================================
# FREQUENCY DISTRIBUTION BY TRADE SIZE (Nº TRADES) + TABLA
# ============================================================

trades_side_bucket = (
    df_vol.groupby(["vol_bucket", "buyOrSell"])
          .size()
          .unstack("buyOrSell")
          .reindex(bin_labels)
          .fillna(0)
          .reset_index()
          .rename(columns={"vol_bucket": "bucket"})
)

trades_side_bucket["buy_trades"]  = trades_side_bucket.get("Buy", 0).astype(int)
trades_side_bucket["sell_trades"] = trades_side_bucket.get("Sell", 0).astype(int)
trades_side_bucket["total_trades"] = (
    trades_side_bucket["buy_trades"] + trades_side_bucket["sell_trades"]
)

total_trades_all = trades_side_bucket["total_trades"].sum()

trades_side_bucket["total_share"] = np.where(
    total_trades_all > 0,
    trades_side_bucket["total_trades"] * 100.0 / total_trades_all,
    0.0,
)
trades_side_bucket["buy_share"] = np.where(
    trades_side_bucket["total_trades"] > 0,
    trades_side_bucket["buy_trades"] * 100.0 / trades_side_bucket["total_trades"],
    0.0,
)
trades_side_bucket["sell_share"] = np.where(
    trades_side_bucket["total_trades"] > 0,
    trades_side_bucket["sell_trades"] * 100.0 / trades_side_bucket["total_trades"],
    0.0,
)

# para apilar barras
trades_side_bucket["trades_top"] = (
    trades_side_bucket["buy_trades"] + trades_side_bucket["sell_trades"]
)

# strings para tabla de FRECUENCIA
trades_side_bucket["total_trades_str"] = trades_side_bucket["total_trades"].map(lambda x: f"{x:,.0f}")
trades_side_bucket["total_share_str"]  = trades_side_bucket["total_share"].map(lambda x: f"{x:,.2f}%")
trades_side_bucket["buy_trades_str"]   = trades_side_bucket["buy_trades"].map(lambda x: f"{x:,.0f}")
trades_side_bucket["sell_trades_str"]  = trades_side_bucket["sell_trades"].map(lambda x: f"{x:,.0f}")
trades_side_bucket["buy_share_str"]    = trades_side_bucket["buy_share"].map(lambda x: f"{x:,.2f}%")
trades_side_bucket["sell_share_str"]   = trades_side_bucket["sell_share"].map(lambda x: f"{x:,.2f}%")

trades_side_bucket["total_trades_is_max"] = (
    trades_side_bucket["total_trades"] == trades_side_bucket["total_trades"].max()
)
trades_side_bucket["total_trades_is_min"] = (
    trades_side_bucket["total_trades"] == trades_side_bucket["total_trades"].min()
)
trades_side_bucket["share_trades_is_max"] = (
    trades_side_bucket["total_share"] == trades_side_bucket["total_share"].max()
)
trades_side_bucket["share_trades_is_min"] = (
    trades_side_bucket["total_share"] == trades_side_bucket["total_share"].min()
)

source_tdist = ColumnDataSource(trades_side_bucket)

p_trades_dist = figure(
    x_range=bin_labels,
    width=650,
    height=350,
    title="Trade Count Distribution by Trade Size (Buckets by |vol| per trade)",
    toolbar_location="right",
    tools="pan,box_zoom,reset,save",
)

r_buy_td = p_trades_dist.vbar(
    x="bucket",
    top="buy_trades",
    width=0.8,
    source=source_tdist,
    fill_color="#1f77b4",
    legend_label="Buy trades",
)

r_sell_td = p_trades_dist.vbar(
    x="bucket",
    bottom="buy_trades",
    top="trades_top",
    width=0.8,
    source=source_tdist,
    fill_color="#ff7f0e",
    legend_label="Sell trades",
)

p_trades_dist.yaxis.axis_label = "Number of trades"
p_trades_dist.yaxis.formatter = NumeralTickFormatter(format="0a")
p_trades_dist.xaxis.major_label_orientation = 0.85

legend_td = p_trades_dist.legend[0]
p_trades_dist.add_layout(legend_td, "right")
p_trades_dist.legend.click_policy = "hide"

hover_trades_dist = HoverTool(
    renderers=[r_buy_td, r_sell_td],
    tooltips=[
        ("Bucket (by |vol|)", "@bucket"),
        ("Total trades", "@total_trades{0,0}"),
        ("Share of trades", "@total_share{0,0.00}%"),
        ("Buy trades", "@buy_trades{0,0}"),
        ("Sell trades", "@sell_trades{0,0}"),
        ("Buy % in bucket", "@buy_share{0,0.00}%"),
        ("Sell % in bucket", "@sell_share{0,0.00}%"),
    ],
)
p_trades_dist.add_tools(hover_trades_dist)

# --------- Tabla de FRECUENCIA ----------
template_total_trades_b = """
<% var zebra = (index % 2 === 0) ? '#ffffff' : '#f7f7f7';
   var bg = zebra;
   if (total_trades_is_max) { bg = '#e0f2ff'; }
   else if (total_trades_is_min) { bg = '#ffeaea'; } %>
<div style="background-color:<%= bg %>; text-align:right;">
  <%= total_trades_str %>
</div>
"""

template_total_share_tr_b = """
<% var zebra = (index % 2 === 0) ? '#ffffff' : '#f7f7f7';
   var bg = zebra;
   if (share_trades_is_max) { bg = '#e0f2ff'; }
   else if (share_trades_is_min) { bg = '#ffeaea'; } %>
<div style="background-color:<%= bg %>; text-align:right;">
  <%= total_share_str %>
</div>
"""

template_buy_trades_b = """
<% var zebra = (index % 2 === 0) ? '#ffffff' : '#f7f7f7'; %>
<div style="background-color:<%= zebra %>; text-align:right;">
  <%= buy_trades_str %>
</div>
"""

template_sell_trades_b = """
<% var zebra = (index % 2 === 0) ? '#ffffff' : '#f7f7f7'; %>
<div style="background-color:<%= zebra %>; text-align:right;">
  <%= sell_trades_str %>
</div>
"""

template_buy_share_tr_b = """
<% var zebra = (index % 2 === 0) ? '#ffffff' : '#f7f7f7'; %>
<div style="background-color:<%= zebra %>; text-align:right;">
  <%= buy_share_str %>
</div>
"""

template_sell_share_tr_b = """
<% var zebra = (index % 2 === 0) ? '#ffffff' : '#f7f7f7'; %>
<div style="background-color:<%= zebra %>; text-align:right;">
  <%= sell_share_str %>
</div>
"""

columns_tdist = [
    TableColumn(field="bucket",           title="Bucket (by |vol|)",
                formatter=StringFormatter(text_align="center")),
    TableColumn(field="total_trades_str", title="Total trades",
                formatter=HTMLTemplateFormatter(template=template_total_trades_b)),
    TableColumn(field="total_share_str",  title="Share of trades",
                formatter=HTMLTemplateFormatter(template=template_total_share_tr_b)),
    TableColumn(field="buy_trades_str",   title="Buy trades",
                formatter=HTMLTemplateFormatter(template=template_buy_trades_b)),
    TableColumn(field="sell_trades_str",  title="Sell trades",
                formatter=HTMLTemplateFormatter(template=template_sell_trades_b)),
    TableColumn(field="buy_share_str",    title="Buy %",
                formatter=HTMLTemplateFormatter(template=template_buy_share_tr_b)),
    TableColumn(field="sell_share_str",   title="Sell %",
                formatter=HTMLTemplateFormatter(template=template_sell_share_tr_b)),
]

table_tdist = DataTable(
    source=source_tdist,
    columns=columns_tdist,
    width=450,
    height=350,
    index_position=None,
)



# ========= Intraday profile: volume & trades by hour =========
df_hour = df.copy()
df_hour["tradeTime"] = pd.to_datetime(df_hour["tradeTime"])
df_hour["date"] = df_hour["tradeTime"].dt.normalize()
df_hour["hour"] = df_hour["tradeTime"].dt.hour
df_hour["abs_vol"] = df_hour["vol"].abs()

daily_hour = (
    df_hour.groupby(["date", "hour"])
           .agg(volume=("abs_vol", "sum"),
                trades=("isin", "count"))
           .reset_index()
)

hour_stats = (
    daily_hour.groupby("hour")
              .agg(volume_mean=("volume", "mean"),
                   volume_std=("volume", "std"),
                   trades_mean=("trades", "mean"),
                   trades_std=("trades", "std"))
              .reset_index()
)

hour_stats[["volume_std", "trades_std"]] = hour_stats[["volume_std", "trades_std"]].fillna(0.0)

# >>> Bucket tipo '08:00-09:00'
hour_stats["hour_label"] = hour_stats["hour"].apply(
    lambda h: f"{h:02d}:00-{(h+1) % 24:02d}:00"
)

total_vol_mean = hour_stats["volume_mean"].sum()
total_tr_mean  = hour_stats["trades_mean"].sum()

hour_stats["volume_share"] = np.where(
    total_vol_mean > 0,
    hour_stats["volume_mean"] * 100.0 / total_vol_mean, 0.0
)
hour_stats["trades_share"] = np.where(
    total_tr_mean > 0,
    hour_stats["trades_mean"] * 100.0 / total_tr_mean, 0.0
)

hour_stats["vol_y0"] = hour_stats["volume_mean"] - hour_stats["volume_std"]
hour_stats["vol_y1"] = hour_stats["volume_mean"] + hour_stats["volume_std"]
hour_stats["tr_y0"]  = hour_stats["trades_mean"] - hour_stats["trades_std"]
hour_stats["tr_y1"]  = hour_stats["trades_mean"] + hour_stats["trades_std"]

hour_stats["volume_mean_str"]   = hour_stats["volume_mean"].apply(fmt_num)
hour_stats["volume_share_str"]  = hour_stats["volume_share"].apply(lambda x: f"{x:,.2f}%")
hour_stats["trades_mean_str"]   = hour_stats["trades_mean"].apply(fmt_num)
hour_stats["trades_share_str"]  = hour_stats["trades_share"].apply(lambda x: f"{x:,.2f}%")

for col in ["volume_mean", "volume_share", "trades_mean", "trades_share"]:
    hour_stats[f"{col}_is_max"] = hour_stats[col] == hour_stats[col].max()
    hour_stats[f"{col}_is_min"] = hour_stats[col] == hour_stats[col].min()

source_hour = ColumnDataSource(hour_stats)
hour_labels = hour_stats["hour_label"].tolist()
fmt_axis_hour = NumeralTickFormatter(format="0a")

# --------- Gráfico 1: Volumen intradía ----------
p_hour_vol = figure(
    x_range=hour_labels,
    width=650,
    height=350,
    title="Intraday Profile – Average Daily Volume by Hour",
    toolbar_location="right",
    tools="pan,box_zoom,reset,save",
)

# barras de ±1σ
p_hour_vol.segment(
    "hour_label", "vol_y0",
    "hour_label", "vol_y1",
    source=source_hour,
    line_width=2,
    line_color="#6baed6",
)

r_hour_vol = p_hour_vol.line(
    "hour_label",
    "volume_mean",
    source=source_hour,
    line_width=2,
    line_color="black",
)

p_hour_vol.scatter("hour_label", "volume_mean", source=source_hour, size=8)

p_hour_vol.yaxis.formatter = fmt_axis_hour
p_hour_vol.xaxis.major_label_orientation = 0.9

p_hour_vol.add_tools(
    HoverTool(
        renderers=[r_hour_vol],
        tooltips=[
            ("Hour bucket", "@hour_label"),
            ("Average volume", "@volume_mean{0,0.00}"),
            ("Std dev", "@volume_std{0,0.00}"),
            ("Share of volume", "@volume_share{0,0.00}%"),
        ],
    )
)

hour_vol_note = Div(
    width=650,
    text=(
        "Note. The line shows the average daily trading volume for each hour bucket "
        "(e.g. 08:00-09:00). Vertical bars indicate ±1 standard deviation across "
        "trading days."
    ),
)

template_hour_vol = """
<%
  var zebra = (index % 2 === 0) ? "#ffffff" : "#f7f7f7";
  var bg = zebra;
  if (volume_mean_is_max) { bg = "#e0f2ff"; }
  else if (volume_mean_is_min) { bg = "#ffeaea"; }
%>
<div style="background-color:<%= bg %>; text-align:right;">
  <%= volume_mean_str %>
</div>
"""

template_hour_vol_share = """
<%
  var zebra = (index % 2 === 0) ? "#ffffff" : "#f7f7f7";
  var bg = zebra;
  if (volume_share_is_max) { bg = "#e0f2ff"; }
  else if (volume_share_is_min) { bg = "#ffeaea"; }
%>
<div style="background-color:<%= bg %>; text-align:right;">
  <%= volume_share_str %>
</div>
"""

columns_hour_vol = [
    TableColumn(field="hour_label",       title="Hour bucket",
                formatter=StringFormatter(text_align="center")),
    TableColumn(field="volume_mean_str",  title="Avg volume",
                formatter=HTMLTemplateFormatter(template=template_hour_vol)),
    TableColumn(field="volume_share_str", title="Share of volume",
                formatter=HTMLTemplateFormatter(template=template_hour_vol_share)),
]

table_hour_vol = DataTable(
    source=source_hour,
    columns=columns_hour_vol,
    width=450,
    height=350,
    index_position=None,
)

# --------- Gráfico 2: # Trades intradía ----------
p_hour_trades = figure(
    x_range=hour_labels,
    width=650,
    height=350,
    title="Intraday Profile – Average Daily Number of Trades by Hour",
    toolbar_location="right",
    tools="pan,box_zoom,reset,save",
)

r_hour_tr = p_hour_trades.vbar(
    x="hour_label",
    top="trades_mean",
    width=0.8,
    source=source_hour,
)

p_hour_trades.yaxis.formatter = fmt_axis_hour
p_hour_trades.xaxis.major_label_orientation = 0.9

p_hour_trades.add_tools(
    HoverTool(
        renderers=[r_hour_tr],
        tooltips=[
            ("Hour bucket", "@hour_label"),
            ("Average trades", "@trades_mean{0,0.00}"),
            ("Std dev", "@trades_std{0,0.00}"),
            ("Share of trades", "@trades_share{0,0.00}%"),
        ],
    )
)

hour_trades_note = Div(
    width=650,
    text=(
        "Note. The bars show the average daily number of trades for each hour bucket "
        "(e.g. 08:00-09:00)."
    ),
)

template_hour_tr = """
<%
  var zebra = (index % 2 === 0) ? "#ffffff" : "#f7f7f7";
  var bg = zebra;
  if (trades_mean_is_max) { bg = "#e0f2ff"; }
  else if (trades_mean_is_min) { bg = "#ffeaea"; }
%>
<div style="background-color:<%= bg %>; text-align:right;">
  <%= trades_mean_str %>
</div>
"""

template_hour_tr_share = """
<%
  var zebra = (index % 2 === 0) ? "#ffffff" : "#f7f7f7";
  var bg = zebra;
  if (trades_share_is_max) { bg = "#e0f2ff"; }
  else if (trades_share_is_min) { bg = "#ffeaea"; }
%>
<div style="background-color:<%= bg %>; text-align:right;">
  <%= trades_share_str %>
</div>
"""

columns_hour_tr = [
    TableColumn(field="hour_label",       title="Hour bucket",
                formatter=StringFormatter(text_align="center")),
    TableColumn(field="trades_mean_str",  title="Avg # trades",
                formatter=HTMLTemplateFormatter(template=template_hour_tr)),
    TableColumn(field="trades_share_str", title="Share of trades",
                formatter=HTMLTemplateFormatter(template=template_hour_tr_share)),
]

table_hour_trades = DataTable(
    source=source_hour,
    columns=columns_hour_tr,
    width=450,
    height=350,
    index_position=None,
)


# ============================================================
# MONTHLY VOLUME BY BUCKET (GROSS & NET) + GRÁFICO + TABLA
# ============================================================

# (mantengo tu bloque completo aquí sin cambios adicionales, porque no afecta
# a la parte de los dos primeros gráficos / tabla única)

# 1) Agregamos por (month, bucket): volumen bruto y neto por bucket
monthly_bucket = (
    df_vol.groupby(["month", "vol_bucket"])
          .agg(
              gross_bucket_vol=("abs_vol", "sum"),  # suma de |vol|
              net_bucket_vol=("vol", "sum"),        # suma de vol con signo
          )
          .reset_index()
          .rename(columns={"vol_bucket": "bucket"})
)

# Añadimos etiquetas de mes y factor categórico para el eje X (reusamos 'monthly')
monthly_bucket = monthly_bucket.merge(
    monthly[["month", "month_label", "month_factor"]],
    on="month",
    how="left",
)

# ============================================================
# GRÁFICO: NET VOLUME POR BUCKET A TRAVÉS DEL TIEMPO
# ============================================================

net_pivot = (
    monthly_bucket.pivot_table(
        index="month", columns="bucket", values="net_bucket_vol", fill_value=0.0
    )
    .reindex(monthly["month"])
)

x_factors = monthly["month_factor"].tolist()
x_labels  = monthly["month_label"].tolist()

p_net_bucket = figure(
    x_range=month_xrange,
    width=900,
    height=350,
    title="Monthly Net Volume by Trade Size Bucket (buckets by |vol| per trade)",
    toolbar_location="right",
    tools="pan,box_zoom,reset,save",
)

marker_funcs = [
    p_net_bucket.circle,
    p_net_bucket.square,
    p_net_bucket.triangle,
    p_net_bucket.diamond,
    p_net_bucket.cross,
    p_net_bucket.inverted_triangle,
]

from bokeh.models import Legend, LegendItem

renderers_bucket = []
legend_items = []

for i, bucket_label in enumerate(bin_labels):
    if bucket_label not in net_pivot.columns:
        continue

    y_values = net_pivot[bucket_label].tolist()
    colors = ["green" if v >= 0 else "red" for v in y_values]

    source_b = ColumnDataSource(
        data=dict(
            month_factor=x_factors,
            month_label=x_labels,
            bucket=[bucket_label] * len(x_factors),
            net_bucket_vol=y_values,
            color=colors,
        )
    )

    marker_fn = marker_funcs[i % len(marker_funcs)]

    # MARKER (color dinámico por signo)
    r_marker = marker_fn(
        x="month_factor",
        y="net_bucket_vol",
        source=source_b,
        size=9,
        fill_color="color",
        line_color="color",
    )

    # LÍNEA gris
    r_line = p_net_bucket.line(
        x="month_factor",
        y="net_bucket_vol",
        source=source_b,
        line_width=1.5,
        line_alpha=0.6,
        color="#888888",
    )

    renderers_bucket.append(r_marker)

    legend_items.append(
        LegendItem(
            label=str(bucket_label),
            renderers=[r_marker, r_line],
        )
    )

legend = Legend(items=legend_items, location="center")
p_net_bucket.add_layout(legend, "right")
p_net_bucket.legend.click_policy = "hide"

p_net_bucket.xaxis.major_label_orientation = 0.9
p_net_bucket.yaxis.formatter = fmt_axis_vol
p_net_bucket.yaxis.axis_label = "Net volume"

hover_net_bucket = HoverTool(
    renderers=renderers_bucket,
    tooltips=[
        ("Month", "@month_label"),
        ("Bucket (by |vol|)", "@bucket"),
        ("Net volume", "@net_bucket_vol{0,0.00}"),
    ],
)
p_net_bucket.add_tools(hover_net_bucket)


# ============================================================
# TABLA PIVOT: BUCKETS x MESES (NET VOLUME) + ALL
# ============================================================

pivot_table = (
    monthly_bucket.pivot_table(
        index="bucket", columns="month_label", values="net_bucket_vol", fill_value=0.0
    )
    .reindex(bin_labels)
)

pivot_table = pivot_table.loc[~pivot_table.index.isna()]

month_cols_order = x_labels
existing_cols = [m for m in month_cols_order if m in pivot_table.columns]
pivot_table = pivot_table[existing_cols]

pivot_table["ALL"] = pivot_table.sum(axis=1)
col_sums = pivot_table.sum(axis=0)
pivot_table.loc["ALL"] = col_sums

row_order = [b for b in bin_labels if b in pivot_table.index] + ["ALL"]
pivot_table = pivot_table.reindex(row_order)

table_data = {"bucket": pivot_table.index.tolist()}
col_order = existing_cols + ["ALL"]

for col in col_order:
    table_data[col] = pivot_table[col].tolist()

source_month_bucket_table = ColumnDataSource(table_data)

template_net_pivot_cell = """
<%
  var v = (value == null ? 0 : value);
  var color = (v >= 0) ? 'green' : 'red';

  // Convertimos a string con 2 decimales
  var s = v.toFixed(2);

  // Insertamos separador de miles (',')
  s = s.replace(/\\B(?=(\\d{3})+(?!\\d))/g, ",");
%>
<div style="text-align:right; color:<%= color %>;">
  <%= s %>
</div>
"""




columns_month_bucket = [
    TableColumn(
        field="bucket",
        title="Bucket (by |vol|)",
        formatter=StringFormatter(text_align="center"),
    )
]

for col in col_order:
    columns_month_bucket.append(
        TableColumn(
            field=col,
            title=col,
            formatter=HTMLTemplateFormatter(template=template_net_pivot_cell),
        )
    )


table_month_bucket = DataTable(
    source=source_month_bucket_table,
    columns=columns_month_bucket,
    width=1400,
    height=400,
    index_position=None,
)


# ============================================================
# LAYOUTS & OUTPUT
# ============================================================

from bokeh.models import TabPanel, Tabs

# BLOQUE 1 + 2: dos primeros gráficos alineados y UNA tabla debajo
layout_with_tables = column(
    row(p_vol_gross, p_vol_net),          # <<< CAMBIO: dos gráficos juntos
    row(table_gross),                     # <<< CAMBIO: una única tabla larga
    row(p_vol_dist, table_vdist, p_trades_dist, table_tdist),
    row(column(p_hour_vol, hour_vol_note), table_hour_vol),
    row(p_net_bucket, table_month_bucket),
)

layout_without_tables = column(
    p_vol_gross,
    p_vol_net,
    p_vol_dist,
    column(p_hour_vol, hour_vol_note),
    row(p_net_bucket, table_month_bucket),
)

tab_full = TabPanel(child=layout_with_tables, title="Full Report (with tables)")
tab_simple = TabPanel(child=layout_without_tables, title="Charts Only")

tabs = Tabs(tabs=[tab_full])  # solo una pestaña, como ya tenías

output_file("trading_report.html", title="Trading Activity Report")
save(tabs)

print("HTML generated: trading_report.html")
